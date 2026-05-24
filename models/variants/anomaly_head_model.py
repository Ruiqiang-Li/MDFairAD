"""Backward-compatible shim re-exporting the four anomaly heads now defined under ``models/variants/anomaly_heads/``."""

from models.variants.anomaly_heads.dominant_head import (
    DominantAnomalyHead,
    DominantAttributeDecoder,
    DominantStructureDecoder,
)
from models.variants.anomaly_heads.vgod_head import (
    CovConv,
    MeanConv,
    VGODAnomalyHead,
    VGODReconDecoder,
    VGODVarianceDecoder,
)
from models.variants.anomaly_heads.conad_head import (
    CONADAnomalyHead,
    CONADAttributeEncoder,
    CONADDiscriminator,
    CONADModel,
    CONADSharedEncoder,
)
from models.variants.anomaly_heads.cola_head import (
    AvgReadout,
    CoLAAnomalyHead,
    CoLADiscriminator,
    CoLAGCN,
    CoLAModel,
    CoLASubgraphBuilder,
    MaxReadout,
    MinReadout,
    WSReadout,
)

__all__ = [
    'DominantAnomalyHead',
    'DominantAttributeDecoder',
    'DominantStructureDecoder',
    'VGODAnomalyHead',
    'VGODReconDecoder',
    'VGODVarianceDecoder',
    'MeanConv',
    'CovConv',
    'CONADAnomalyHead',
    'CONADModel',
    'CONADSharedEncoder',
    'CONADAttributeEncoder',
    'CONADDiscriminator',
    'CoLAAnomalyHead',
    'CoLAModel',
    'CoLAGCN',
    'CoLADiscriminator',
    'CoLASubgraphBuilder',
    'AvgReadout',
    'MaxReadout',
    'MinReadout',
    'WSReadout',
]
