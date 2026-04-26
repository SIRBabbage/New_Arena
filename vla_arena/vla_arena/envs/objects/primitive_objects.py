# Copyright 2025 The VLA-Arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

from robosuite.models.objects import BoxObject

from vla_arena.vla_arena.envs.base_object import register_object


class PrimitiveBoxObject(BoxObject):
    def __init__(
        self,
        name,
        size,
        rgba,
        density=500,
        friction=(1.0, 0.005, 0.0001),
        joints='default',
    ):
        super().__init__(
            name=name,
            size=size,
            rgba=rgba,
            density=density,
            friction=friction,
            joints=joints,
            obj_type='all',
            duplicate_collision_geoms=True,
        )
        self.category_name = '_'.join(
            re.sub(r'([A-Z0-9])', r' \1', self.__class__.__name__).split()
        ).lower()
        self.rotation = (0, 0)
        self.rotation_axis = 'z'
        self.object_properties = {'vis_site_names': {}}


@register_object
class Marker(PrimitiveBoxObject):
    def __init__(self, name='marker', joints='default'):
        super().__init__(
            name=name,
            size=(0.055, 0.008, 0.008),
            rgba=(0.12, 0.38, 0.95, 1.0),
            density=220,
            joints=joints,
        )


@register_object
class Spoon(PrimitiveBoxObject):
    def __init__(self, name='spoon', joints='default'):
        super().__init__(
            name=name,
            size=(0.065, 0.016, 0.005),
            rgba=(0.74, 0.74, 0.78, 1.0),
            density=320,
            joints=joints,
        )


@register_object
class GreenCube(PrimitiveBoxObject):
    def __init__(self, name='green_cube', joints='default'):
        super().__init__(
            name=name,
            size=(0.022, 0.022, 0.022),
            rgba=(0.18, 0.72, 0.28, 1.0),
            density=450,
            joints=joints,
        )


@register_object
class YellowCube(PrimitiveBoxObject):
    def __init__(self, name='yellow_cube', joints='default'):
        super().__init__(
            name=name,
            size=(0.022, 0.022, 0.022),
            rgba=(0.94, 0.82, 0.18, 1.0),
            density=450,
            joints=joints,
        )
