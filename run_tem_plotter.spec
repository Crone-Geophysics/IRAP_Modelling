# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['run_tem_plotter.py'],
             pathex=['C:\\Users\\Mortulo\\PycharmProjects\\IRAP_Modelling'],
             binaries=[],
             datas=[(r'src/ui/*.ui', 'ui'),
					(r'src/ui/icons', r'ui/icons')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='TEMPlotter',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
		  icon='tem_plotter.ico',
          console=False)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='TEM Plotter v0.0')
