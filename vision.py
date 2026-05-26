"""
Vision — sensory grounding for the brain.

Implements:
  - Retinal center-surround processing (ON/OFF ganglion cells)
  - V1 simple cells (Gabor filter bank at multiple orientations/scales)
  - V1 complex cells (energy model — pooling over phase)
  - Bottom-up saliency map (for attention guidance)
"""

import numpy as np
from scipy.ndimage import convolve


def gabor2d(width, theta, freq=0.15, sigma=4.0, psi=0.0):
    """Generate a 2D Gabor filter at orientation theta.

    Args:
        width: filter size in pixels (odd)
        theta: orientation in radians
        freq: spatial frequency
        sigma: Gaussian envelope width
        psi: phase offset
    Returns:
        (width, width) numpy array
    """
    half = width // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gauss = np.exp(-0.5 * (x_theta**2 + y_theta**2) / sigma**2)
    sinusoid = np.cos(2 * np.pi * freq * x_theta + psi)
    return gauss * sinusoid


class Retina:
    """Retinal processing: center-surround contrast enhancement.

    Simulates ON-center / OFF-surround and OFF-center / ON-surround
    ganglion cells using Difference-of-Gaussians (DoG).
    """

    def __init__(self, kernel_size=7, center_sigma=1.0, surround_sigma=3.0):
        self.kernel_size = kernel_size
        self.center_sigma = center_sigma
        self.surround_sigma = surround_sigma
        self._build_kernels()

    def _build_kernels(self):
        half = self.kernel_size // 2
        y, x = np.mgrid[-half:half+1, -half:half+1]
        r2 = x**2 + y**2
        center = np.exp(-r2 / (2 * self.center_sigma**2))
        center /= center.sum()
        surround = np.exp(-r2 / (2 * self.surround_sigma**2))
        surround /= surround.sum()
        self.on_kernel = center - 0.5 * surround
        self.off_kernel = -self.on_kernel

    def process(self, image):
        """Process a grayscale image through retina.

        Args:
            image: (H, W) numpy array, values [0, 1]
        Returns:
            on_response: (H, W) ON-cell responses
            off_response: (H, W) OFF-cell responses
        """
        on_resp = convolve(image, self.on_kernel, mode='reflect')
        off_resp = convolve(image, self.off_kernel, mode='reflect')
        on_resp = np.maximum(0, on_resp)
        off_resp = np.maximum(0, off_resp)
        return on_resp, off_resp


class V1SimpleCells:
    """V1 simple cells — Gabor filter bank.

    Responds to oriented edges at specific positions and phases.
    Each cell has a preferred orientation, scale, and phase.
    """

    def __init__(self, filter_size=11, num_orientations=8, num_scales=3,
                 num_phases=2, input_height=64, input_width=64):
        self.filter_size = filter_size
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.num_phases = num_phases
        self.input_height = input_height
        self.input_width = input_width

        orientations = np.linspace(0, np.pi, num_orientations, endpoint=False)
        scales = np.linspace(3.0, 6.0, num_scales)
        phases = np.linspace(0, np.pi, num_phases, endpoint=False)

        self.filters = []
        self.filter_params = []
        for theta in orientations:
            for sigma in scales:
                for psi in phases:
                    f = gabor2d(filter_size, theta, freq=1.0/(sigma*2),
                                sigma=sigma, psi=psi)
                    f -= f.mean()
                    self.filters.append(f)
                    self.filter_params.append((theta, sigma, psi))
        self.num_filters = len(self.filters)
        self.output_shape = (self.num_filters,
                             input_height - filter_size + 1,
                             input_width - filter_size + 1)

    def process(self, image):
        """Convolve image with all Gabor filters.

        Args:
            image: (H, W) grayscale image
        Returns:
            responses: (num_filters, H-f+1, W-f+1) filter responses
        """
        out_h = image.shape[0] - self.filter_size + 1
        out_w = image.shape[1] - self.filter_size + 1
        responses = np.zeros((self.num_filters, out_h, out_w))
        for i, f in enumerate(self.filters):
            full = convolve(image, f, mode='reflect')
            responses[i] = full[:out_h, :out_w]
        responses = np.maximum(0, responses)  # half-wave rectification
        return responses

    def flatten(self, responses):
        """Flatten V1 responses into a 1D feature vector for cortical input."""
        return responses.reshape(self.num_filters, -1).mean(axis=1)


class V1ComplexCells:
    """V1 complex cells — energy model.

    Pools over quadrature phase pairs to achieve phase-invariant
    orientation selectivity.  |response| = sqrt(sin² + cos²).
    """

    def __init__(self, filter_size=11, num_orientations=8, num_scales=3):
        self.filter_size = filter_size
        self.num_orientations = num_orientations
        self.num_scales = num_scales

        orientations = np.linspace(0, np.pi, num_orientations, endpoint=False)
        scales = np.linspace(3.0, 6.0, num_scales)

        self.sin_filters = []
        self.cos_filters = []
        for theta in orientations:
            for sigma in scales:
                cos_f = gabor2d(filter_size, theta, freq=1.0/(sigma*2),
                                sigma=sigma, psi=0.0)
                sin_f = gabor2d(filter_size, theta, freq=1.0/(sigma*2),
                                sigma=sigma, psi=np.pi/2)
                cos_f -= cos_f.mean()
                sin_f -= sin_f.mean()
                self.cos_filters.append(cos_f)
                self.sin_filters.append(sin_f)
        self.num_filters = len(self.cos_filters)

    def process(self, image):
        """Compute energy (phase-invariant) responses.

        Returns:
            energy: (num_filters, H-f+1, W-f+1) energy responses
        """
        out_h = image.shape[0] - self.filter_size + 1
        out_w = image.shape[1] - self.filter_size + 1
        responses = np.zeros((self.num_filters, out_h, out_w))
        for i in range(self.num_filters):
            cos_full = convolve(image, self.cos_filters[i], mode='reflect')
            sin_full = convolve(image, self.sin_filters[i], mode='reflect')
            responses[i] = np.sqrt(cos_full[:out_h, :out_w]**2 +
                                   sin_full[:out_h, :out_w]**2)
        return responses

    def flatten(self, responses):
        """Flatten complex cell responses to a 1D vector."""
        return responses.reshape(self.num_filters, -1).mean(axis=1)


class SaliencyMap:
    """Bottom-up visual saliency based on center-surround contrast.

    Used as input to the thalamic TRN for attentional gating.
    """

    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size
        half = kernel_size // 2
        y, x = np.mgrid[-half:half+1, -half:half+1]
        center = np.exp(-(x**2 + y**2) / (2 * 1.0**2))
        center /= center.sum()
        surround = np.exp(-(x**2 + y**2) / (2 * 3.0**2))
        surround /= surround.sum()
        self.dog = center - surround

    def compute(self, feature_map):
        """Compute saliency from a feature map.

        Returns:
            saliency: (H, W) saliency values, normalized [0, 1]
        """
        if feature_map.ndim == 3:
            feature_map = np.mean(feature_map, axis=0)
        contrast = np.abs(convolve(feature_map, self.dog, mode='reflect'))
        sal = contrast - contrast.mean()
        sal = np.maximum(0, sal)
        sal = sal / (sal.max() + 1e-12)
        return sal


class VisualCortex:
    """Full visual processing pipeline: Retina → V1 simple → V1 complex.

    Provides a cortical-ready feature vector and a saliency map
    that can be routed through the thalamus to neocortical columns.
    """

    def __init__(self, input_height=64, input_width=64):
        self.input_height = input_height
        self.input_width = input_width
        self.retina = Retina()
        self.simple = V1SimpleCells(input_height=input_height,
                                    input_width=input_width)
        self.complex = V1ComplexCells()

    def process(self, image):
        """Full visual processing pipeline.

        Args:
            image: (H, W) or (H, W, 3) image, values [0, 1]
        Returns:
            feature_vector: 1D array for cortical input
            saliency: (H', W') saliency map for attention
        """
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        # Retina: contrast enhancement
        on_resp, off_resp = self.retina.process(image)
        retina_out = on_resp - off_resp

        # V1 simple cells: oriented edge detection
        simple_resp = self.simple.process(retina_out)
        feature_vector = self.simple.flatten(simple_resp)

        # V1 complex cells: phase-invariant energy
        complex_resp = self.complex.process(retina_out)
        saliency = SaliencyMap().compute(
            complex_resp.reshape(-1, complex_resp.shape[-1])
        )

        return feature_vector, saliency

    @property
    def feature_dim(self):
        """Dimension of the feature vector produced by this V1."""
        return self.simple.num_filters


def test_vision():
    """Quick smoke test — verify processing pipeline."""
    np.random.seed(42)

    # Create synthetic image (grating)
    h, w = 64, 64
    y, x = np.mgrid[:h, :w]
    grating = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * (x * np.cos(0.5) + y * np.sin(0.5)))
    grating += np.random.randn(h, w) * 0.05
    grating = np.clip(grating, 0, 1)

    vc = VisualCortex()
    feat, sal = vc.process(grating)
    assert feat.ndim == 1
    assert feat.shape[0] > 0
    print(f"  ✓ VisualCortex: {feat.shape[0]} features, saliency {sal.shape}")

    # Test retina
    ret = Retina()
    on_r, off_r = ret.process(grating)
    assert on_r.shape == (h, w)
    print(f"  ✓ Retina: ON/OFF channels ({on_r.shape})")

    # Test V1 simple
    v1s = V1SimpleCells()
    sr = v1s.process(grating)
    assert sr.shape[0] == v1s.num_filters
    print(f"  ✓ V1Simple: {v1s.num_filters} filters, responses {sr.shape}")

    # Flatten test
    flat = v1s.flatten(sr)
    assert flat.shape == (v1s.num_filters,)
    print(f"  ✓ V1 flatten: {flat.shape}")

    # Test complex cells
    v1c = V1ComplexCells()
    cr = v1c.process(grating)
    assert cr.shape[0] == v1c.num_filters
    print(f"  ✓ V1Complex: {v1c.num_filters} energy units")

    print("  ✓ Vision module OK")


if __name__ == "__main__":
    test_vision()
