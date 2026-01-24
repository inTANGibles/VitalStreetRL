"""业态类型集合：管理业态类型定义和属性"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class BusinessCategory(Enum):
    """业态大类"""
    DINING = "dining"                      # 餐饮
    RETAIL = "retail"                      # 零售
    CULTURAL_EXPERIENCE = "cultural"       # 文化体验
    SERVICE = "service"                    # 服务（办公、中医理疗等）
    LEISURE_ENTERTAINMENT = "leisure"      # 休闲娱乐（酒吧、休闲娱乐等）
    RESIDENTIAL = "residential"            # 居住（民居）
    SUPPORTING_FACILITIES = "supporting"   # 配套设施
    UNDEFINED = "undefined"                # 未定义


@dataclass
class BusinessType:
    """业态类型定义"""
    type_id: str
    name: str
    category: BusinessCategory
    capacity_range: tuple  # (min, max) 容量范围（面积，平方米）
    compatibility: Dict[str, float]  # 与其他业态的兼容度（0-1）


class BusinessTypeCollection:
    """
    业态类型集合
    
    管理所有可用的业态类型定义，提供：
    - 业态类型查询
    - 替换成本查询
    - 兼容性查询
    """
    
    def __init__(self):
        self.__business_types: Dict[str, BusinessType] = {}
        self._init_default_types()
    
    def _init_default_types(self):
        """初始化默认业态类型（大类）"""
        # 餐饮类
        self.add_business_type(BusinessType(
            type_id="restaurant",
            name="餐厅",
            category=BusinessCategory.DINING,
            capacity_range=(50, 300),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="tea_beverage",
            name="茶饮",
            category=BusinessCategory.DINING,
            capacity_range=(20, 100),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="bakery",
            name="烘焙",
            category=BusinessCategory.DINING,
            capacity_range=(30, 150),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="coffee_shop",
            name="咖啡厅",
            category=BusinessCategory.DINING,
            capacity_range=(30, 200),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="snack_bar",
            name="小吃店",
            category=BusinessCategory.DINING,
            capacity_range=(15, 80),
            compatibility={}
        ))
        
        # 零售类
        self.add_business_type(BusinessType(
            type_id="convenience_store",
            name="便利店",
            category=BusinessCategory.RETAIL,
            capacity_range=(30, 150),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="clothing_store",
            name="服装店",
            category=BusinessCategory.RETAIL,
            capacity_range=(40, 200),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="handicraft_store",
            name="工艺品店",
            category=BusinessCategory.RETAIL,
            capacity_range=(30, 150),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="food_store",
            name="食品店",
            category=BusinessCategory.RETAIL,
            capacity_range=(30, 120),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="bookstore",
            name="书店",
            category=BusinessCategory.RETAIL,
            capacity_range=(50, 300),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="specialty_store",
            name="特产店",
            category=BusinessCategory.RETAIL,
            capacity_range=(30, 150),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="jewelry_store",
            name="珠宝店",
            category=BusinessCategory.RETAIL,
            capacity_range=(40, 200),
            compatibility={}
        ))
        
        # 文化体验类
        self.add_business_type(BusinessType(
            type_id="intangible_cultural_heritage",
            name="非遗文创",
            category=BusinessCategory.CULTURAL_EXPERIENCE,
            capacity_range=(50, 300),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="experience_hall",
            name="体验馆",
            category=BusinessCategory.CULTURAL_EXPERIENCE,
            capacity_range=(100, 500),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="cultural_academy",
            name="文化书院",
            category=BusinessCategory.CULTURAL_EXPERIENCE,
            capacity_range=(150, 800),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="cultural_exhibition",
            name="文化展示交流",
            category=BusinessCategory.CULTURAL_EXPERIENCE,
            capacity_range=(100, 600),
            compatibility={}
        ))
        
        # 服务类
        self.add_business_type(BusinessType(
            type_id="office",
            name="办公",
            category=BusinessCategory.SERVICE,
            capacity_range=(50, 500),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="tcm_therapy",
            name="中医理疗",
            category=BusinessCategory.SERVICE,
            capacity_range=(40, 200),
            compatibility={}
        ))
        
        # 休闲娱乐类
        self.add_business_type(BusinessType(
            type_id="bar",
            name="酒吧",
            category=BusinessCategory.LEISURE_ENTERTAINMENT,
            capacity_range=(50, 300),
            compatibility={}
        ))
        
        self.add_business_type(BusinessType(
            type_id="leisure_entertainment",
            name="休闲娱乐",
            category=BusinessCategory.LEISURE_ENTERTAINMENT,
            capacity_range=(80, 400),
            compatibility={}
        ))
        
        # 居住类
        self.add_business_type(BusinessType(
            type_id="residential",
            name="民居",
            category=BusinessCategory.RESIDENTIAL,
            capacity_range=(60, 200),
            compatibility={}
        ))
        
        # 配套设施类
        self.add_business_type(BusinessType(
            type_id="supporting_facilities",
            name="配套设施",
            category=BusinessCategory.SUPPORTING_FACILITIES,
            capacity_range=(20, 200),
            compatibility={}
        ))
        
        # 初始化兼容性矩阵（简化版，可根据实际需求调整）
        self._init_compatibility_matrix()
    
    def _init_compatibility_matrix(self):
        """初始化业态兼容性矩阵"""
        # 餐饮与零售兼容
        dining_types = ["restaurant", "tea_beverage", "bakery", "coffee_shop", "snack_bar"]
        retail_types = ["convenience_store", "clothing_store", "handicraft_store", 
                       "food_store", "bookstore", "specialty_store", "jewelry_store"]
        
        for dining in dining_types:
            for retail in retail_types:
                self._set_compatibility(dining, retail, 0.7)
        
        # 文化体验与餐饮、零售兼容
        cultural_types = ["intangible_cultural_heritage", "experience_hall", 
                         "cultural_academy", "cultural_exhibition"]
        for cultural in cultural_types:
            for dining in dining_types:
                self._set_compatibility(cultural, dining, 0.8)
            for retail in retail_types:
                self._set_compatibility(cultural, retail, 0.75)
        
        # 休闲娱乐与餐饮兼容
        leisure_types = ["bar", "leisure_entertainment"]
        for leisure in leisure_types:
            for dining in dining_types:
                self._set_compatibility(leisure, dining, 0.8)
    
    def _set_compatibility(self, type1: str, type2: str, value: float):
        """设置两个业态的兼容性"""
        business1 = self.get_business_type(type1)
        business2 = self.get_business_type(type2)
        if business1 and business2:
            business1.compatibility[type2] = value
            business2.compatibility[type1] = value
    
    def add_business_type(self, business_type: BusinessType):
        """添加业态类型"""
        self.__business_types[business_type.type_id] = business_type
    
    def get_business_type(self, type_id: str) -> Optional[BusinessType]:
        """获取业态类型"""
        return self.__business_types.get(type_id)
    
    def get_all_types(self) -> List[BusinessType]:
        """获取所有业态类型"""
        return list(self.__business_types.values())
    
    def get_types_by_category(self, category: BusinessCategory) -> List[BusinessType]:
        """按类别获取业态类型"""
        return [bt for bt in self.__business_types.values() if bt.category == category]
    
    def get_compatibility(self, type1: str, type2: str) -> float:
        """获取两个业态的兼容性"""
        business1 = self.get_business_type(type1)
        business2 = self.get_business_type(type2)
        if business1 and business2:
            return business1.compatibility.get(type2, 0.5)  # 默认0.5
        return 0.5
