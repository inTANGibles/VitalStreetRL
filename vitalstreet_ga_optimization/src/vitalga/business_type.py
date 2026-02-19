"""业态类型：GA 动作解码用（仅保留 BusinessCategory 与最小 BusinessTypeCollection）"""
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


class BusinessCategory(Enum):
    """业态大类"""
    DINING = "dining"
    RETAIL = "retail"
    CULTURAL_EXPERIENCE = "cultural"
    SERVICE = "service"
    LEISURE_ENTERTAINMENT = "leisure"
    RESIDENTIAL = "residential"
    SUPPORTING_FACILITIES = "supporting"
    UNDEFINED = "undefined"


@dataclass
class BusinessType:
    """业态类型定义（最小）"""
    type_id: str
    name: str
    category: BusinessCategory
    capacity_range: tuple
    compatibility: Dict[str, float]


class BusinessTypeCollection:
    """业态类型集合（仅用于 WorldState，解码用 category 列表）"""

    def __init__(self):
        self._types: Dict[str, BusinessType] = {}
        self._init_default_types()

    def _init_default_types(self):
        for cat in BusinessCategory:
            self._types[cat.value] = BusinessType(
                type_id=cat.value,
                name=cat.value,
                category=cat,
                capacity_range=(20, 500),
                compatibility={},
            )

    def get_business_type(self, type_id: str) -> Optional[BusinessType]:
        return self._types.get(type_id)

    def get_all_types(self) -> List[BusinessType]:
        return list(self._types.values())
