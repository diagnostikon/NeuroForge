"""Global Workspace — consciousness-like information broadcast.

The workspace holds a small set of attended information items.
Any module can broadcast to the workspace; the workspace then
broadcasts to all modules. Items compete for access based on
salience, novelty, and emotional valence.
"""

import numpy as np


class WorkspaceItem:
    """A single item in the global workspace."""

    def __init__(self, content, salience=0.5, source='neocortex'):
        self.content = content  # numpy array or token string
        self.salience = salience
        self.source = source
        self.age = 0
        self.broadcasts = 0

    def decay(self, factor=0.8):
        self.salience *= factor
        self.age += 1


class GlobalWorkspace:
    """Global workspace with competition, broadcast, and meta-cognition."""

    def __init__(self, capacity=10, broadcast_decay=0.8):
        self.capacity = capacity
        self.broadcast_decay = broadcast_decay
        self.items = []  # list of WorkspaceItem
        self.broadcast_history = []
        self.global_broadcast = None  # currently broadcasting item
        self.meta_cognition = {
            'total_broadcasts': 0,
            'attention_focus': None,
            'coherence': 0.0,
        }

    def propose(self, content, salience=0.5, source='neocortex'):
        """Propose an item for workspace access."""
        item = WorkspaceItem(content, salience, source)
        self.items.append(item)
        if len(self.items) > self.capacity * 2:
            self._prune()

    def _prune(self):
        """Remove low-salience, old items."""
        self.items.sort(key=lambda x: x.salience * (1.0 - 0.1 * x.age), reverse=True)
        self.items = self.items[:self.capacity]

    def compete(self, attention_modulation=None):
        """Run competition — highest salience item wins broadcast."""
        if not self.items:
            self.global_broadcast = None
            return None

        for item in self.items:
            item.decay(self.broadcast_decay)

        # Compute salience with optional modulation
        if attention_modulation is not None:
            for i, item in enumerate(self.items):
                mod = attention_modulation.get(item.source, 1.0) if isinstance(attention_modulation, dict) else attention_modulation
                item.salience *= mod

        self.items.sort(key=lambda x: x.salience, reverse=True)
        winner = self.items[0]

        if winner.salience > 0.1:
            self.global_broadcast = winner
            winner.broadcasts += 1
            self.broadcast_history.append(winner.content)
            if len(self.broadcast_history) > 100:
                self.broadcast_history.pop(0)
            self.meta_cognition['total_broadcasts'] += 1
            self.meta_cognition['attention_focus'] = winner.source
            self.meta_cognition['coherence'] = winner.salience
            return winner.content
        return None

    def broadcast_to(self, modules):
        """Broadcast the winning item to all registered modules."""
        if self.global_broadcast is None:
            return
        content = self.global_broadcast.content
        for module in modules:
            if hasattr(module, 'receive_broadcast'):
                module.receive_broadcast(content, self.global_broadcast.source)

    def get_current_focus(self):
        """Meta-cognition: what is the system currently aware of?"""
        if self.global_broadcast is None:
            return None
        return {
            'content': self.global_broadcast.content,
            'source': self.global_broadcast.source,
            'salience': self.global_broadcast.salience,
            'broadcasts': self.global_broadcast.broadcasts,
        }

    def report_self(self):
        """Meta-cognition: system introspecting on its own state."""
        return {
            'items_in_workspace': len(self.items),
            'current_focus': self.get_current_focus(),
            'total_broadcasts': self.meta_cognition['total_broadcasts'],
            'coherence': self.meta_cognition['coherence'],
        }
