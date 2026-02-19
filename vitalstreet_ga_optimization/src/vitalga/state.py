"""世界状态与空间单元集合（迁移自原 geo 模块）"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import uuid
import copy
import numpy as np
import geopandas as gpd

from .business_type import BusinessCategory, BusinessTypeCollection

try:
    from shapely.geometry import Polygon
except ImportError:
    Polygon = None

STRtree = None
try:
    from shapely.strtree import STRtree
except ImportError:
    try:
        from shapely.ops import STRtree
    except ImportError:
        pass


class SpaceUnitCollection:
    """空间单元集合：管理片段空间（parcel/unit），GA 动作的直接操作对象"""

    __space_unit_attrs = [
        'uid', 'geometry', 'coords', 'area', 'unit_type', 'enabled', 'protected', 'replaceable',
        'business_type', 'business_category',
        'min_clear_width', 'main_direction', 'adjacent_units', 'flow_prediction'
    ]

    def __init__(self):
        self._SpaceUnitCollection__unit_gdf = gpd.GeoDataFrame(columns=self.__space_unit_attrs)
        self.__uid = uuid.uuid4()

    @staticmethod
    def _create_unit_by_coords(
        coords: np.ndarray,
        unit_type: str = "shop",
        business_type: str = "UNDEFINED",
        business_category: str = None,
        protected: bool = False,
        replaceable: bool = None,
        enabled: bool = None,
    ) -> gpd.GeoDataFrame:
        if Polygon is None:
            raise ImportError("shapely required: pip install shapely")
        geometry = Polygon(coords)
        uid = uuid.uuid4()
        area = geometry.area
        if enabled is None:
            enabled = not protected
        if replaceable is None:
            replaceable = SpaceUnitCollection._infer_replaceable(
                unit_type=unit_type, business_category=business_category, protected=protected
            )
        new_row = {
            'uid': [uid],
            'geometry': [geometry],
            'coords': [coords],
            'area': [area],
            'unit_type': [unit_type],
            'enabled': [enabled],
            'protected': [protected],
            'replaceable': [replaceable],
            'business_type': [business_type] if unit_type == 'shop' else ['N/A'],
            'business_category': [business_category] if business_category else [None],
            'min_clear_width': [0.0] if unit_type == 'public_space' else [None],
            'main_direction': [None],
            'adjacent_units': [[]],
            'flow_prediction': [0.0],
        }
        return gpd.GeoDataFrame(new_row, index=new_row['uid'])

    @staticmethod
    def _infer_replaceable(
        unit_type: str, business_category: str = None, protected: bool = False
    ) -> bool:
        if protected:
            return False
        if unit_type != 'shop':
            return False
        if business_category:
            non_replaceable = ['supporting', 'residential', 'SUPPORTING_FACILITIES', 'RESIDENTIAL']
            if business_category.upper() in [c.upper() for c in non_replaceable]:
                return False
        return True

    def add_space_unit(self, unit, return_uid: bool = True) -> Optional[uuid.UUID]:
        if not self._SpaceUnitCollection__unit_gdf.empty:
            gdf_clean = self._SpaceUnitCollection__unit_gdf.loc[:, ~self._SpaceUnitCollection__unit_gdf.isna().all()]
            unit_clean = unit.loc[:, ~unit.isna().all()]
            self._SpaceUnitCollection__unit_gdf = gpd.pd.concat(
                [gdf_clean, unit_clean], ignore_index=False, sort=False
            )
        else:
            self._SpaceUnitCollection__unit_gdf = unit
        self.__uid = uuid.uuid4()
        if return_uid:
            return unit['uid'].iloc[0] if len(unit) == 1 else unit['uid']

    def get_space_unit_by_uid(self, uid):
        return self._SpaceUnitCollection__unit_gdf.loc[uid]

    def get_all_space_units(self) -> gpd.GeoDataFrame:
        return self._SpaceUnitCollection__unit_gdf

    def get_replaceable_shops(self) -> gpd.GeoDataFrame:
        return self._SpaceUnitCollection__unit_gdf[
            (self._SpaceUnitCollection__unit_gdf['unit_type'] == 'shop')
            & (self._SpaceUnitCollection__unit_gdf['replaceable'] == True)
            & (self._SpaceUnitCollection__unit_gdf['enabled'] == True)
        ]

    def get_shops_for_circulation(self) -> gpd.GeoDataFrame:
        return self._SpaceUnitCollection__unit_gdf[
            (self._SpaceUnitCollection__unit_gdf['unit_type'] == 'shop')
            & (self._SpaceUnitCollection__unit_gdf['enabled'] == True)
        ]

    def get_public_spaces(self) -> gpd.GeoDataFrame:
        return self._SpaceUnitCollection__unit_gdf[
            self._SpaceUnitCollection__unit_gdf['unit_type'] == 'public_space'
        ]

    def get_protected_units(self) -> gpd.GeoDataFrame:
        return self._SpaceUnitCollection__unit_gdf[
            self._SpaceUnitCollection__unit_gdf['protected'] == True
        ]

    def update_business_type(self, uid, new_business_type: str):
        self._SpaceUnitCollection__unit_gdf.loc[uid, 'business_type'] = new_business_type
        self.__uid = uuid.uuid4()

    def convert_to_public_space(self, uid):
        """Shop → Public Space"""
        self._SpaceUnitCollection__unit_gdf.loc[uid, 'unit_type'] = 'public_space'
        self._SpaceUnitCollection__unit_gdf.loc[uid, 'flow_prediction'] = 0.0
        self.__uid = uuid.uuid4()


@dataclass
class WorldState:
    """环境状态表示"""
    space_units: SpaceUnitCollection
    business_types: BusinessTypeCollection
    graph: Any
    budget: float
    constraints: Dict[str, Any]
    step_idx: int = 0
    episode_id: Optional[str] = None

    @classmethod
    def from_geojson(
        cls,
        geojson_path: str,
        budget: float = 1000000.0,
        constraints: Optional[Dict[str, Any]] = None,
        episode_id: Optional[str] = None,
    ) -> 'WorldState':
        geojson_path = Path(geojson_path)
        if not geojson_path.exists():
            raise FileNotFoundError(f"GeoJSON 不存在: {geojson_path}")
        gdf = gpd.read_file(geojson_path)
        space_units = cls._geojson_to_spaceunit_collection(gdf)
        business_types = BusinessTypeCollection()
        if constraints is None:
            constraints = {'protected_zones': [], 'max_budget': budget}
        return cls(
            space_units=space_units,
            business_types=business_types,
            graph=None,
            budget=budget,
            constraints=constraints,
            step_idx=0,
            episode_id=episode_id,
        )

    def copy(self) -> 'WorldState':
        copied_space_units = SpaceUnitCollection()
        original_gdf = self.space_units.get_all_space_units()
        if not original_gdf.empty:
            copied_gdf = original_gdf.copy(deep=True)
            copied_space_units._SpaceUnitCollection__unit_gdf = copied_gdf
        copied_business_types = BusinessTypeCollection()
        copied_constraints = copy.deepcopy(self.constraints)
        return WorldState(
            space_units=copied_space_units,
            business_types=copied_business_types,
            graph=None,
            budget=self.budget,
            constraints=copied_constraints,
            step_idx=self.step_idx,
            episode_id=self.episode_id,
        )

    @staticmethod
    def _geojson_to_spaceunit_collection(gdf: gpd.GeoDataFrame) -> SpaceUnitCollection:
        collection = SpaceUnitCollection()
        for idx, row in gdf.iterrows():
            try:
                geometry = row['geometry']
                if hasattr(geometry, 'exterior'):
                    coords = np.array(geometry.exterior.coords[:-1])
                elif hasattr(geometry, 'geoms'):
                    largest = max(geometry.geoms, key=lambda p: p.area)
                    coords = np.array(largest.exterior.coords[:-1])
                else:
                    continue
                unit_type = row.get('unit_type', 'shop')
                business_type = row.get('business_type', 'UNDEFINED')
                business_category = row.get('business_category', None)
                protected = row.get('protected', False)
                enabled = row.get('enabled', True)
                replaceable = row.get('replaceable', None)
                if business_category is not None and not isinstance(business_category, str):
                    business_category = getattr(business_category, 'value', str(business_category))
                unit_gdf = SpaceUnitCollection._create_unit_by_coords(
                    coords=coords,
                    unit_type=unit_type,
                    business_type=business_type,
                    business_category=business_category,
                    protected=protected,
                    replaceable=replaceable,
                    enabled=enabled,
                )
                if unit_type == 'public_space' and 'flow_prediction' in gdf.columns:
                    flow_val = row.get('flow_prediction', 0.0)
                    unit_gdf.loc[unit_gdf.index[0], 'flow_prediction'] = float(flow_val or 0.0)
                else:
                    unit_gdf.loc[unit_gdf.index[0], 'flow_prediction'] = 0.0
                if unit_type == 'shop':
                    unit_gdf.loc[unit_gdf.index[0], 'flow_prediction'] = 0.0
                collection.add_space_unit(unit_gdf)
            except Exception:
                continue
        all_units = collection.get_all_space_units()
        shop_mask = all_units['unit_type'] == 'shop'
        if shop_mask.any():
            all_units.loc[shop_mask, 'flow_prediction'] = 0.0
            collection._SpaceUnitCollection__unit_gdf = all_units
        return collection
