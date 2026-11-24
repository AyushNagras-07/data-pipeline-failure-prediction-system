"""
attention_gru.py
Direct alias and module export for AttentionGRU and TemporalAttention.
Part of ANNA Course 2307411T - Track 2: Advanced Sequential Models (CO3).
"""

from src.models.attention_model import TemporalAttention, AttentionGRU, AttentionLSTM

__all__ = ["TemporalAttention", "AttentionGRU", "AttentionLSTM"]
