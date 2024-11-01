"""
This script only works in Blender!!
"""
import bpy
import os

folder_path = r"...\ShapeDatabase_Normalized"

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".obj"):
            file_path = os.path.join(root, filename)

            # Import OBJ file
            bpy.ops.wm.obj_import(filepath=file_path)

            # Recalculate normals
            for obj in bpy.context.selected_objects:
                if obj.type == 'MESH':
                    bpy.context.view_layer.objects.active = obj

                    bpy.ops.object.shade_flat()

                    # Recalculate vertice normals
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.select_all(action='SELECT')
                    bpy.ops.mesh.normals_make_consistent(inside=False)

                    # Clean and recalculate split normals
                    bpy.ops.mesh.customdata_custom_splitnormals_clear()
                    bpy.ops.mesh.customdata_custom_splitnormals_add()

                    bpy.ops.object.mode_set(mode='OBJECT')

            # Export
            bpy.ops.wm.obj_export(filepath=file_path, export_selected_objects=True, export_materials=False)

            bpy.ops.object.delete()
            print(f"Processed and saved: {file_path}")