#!/usr/bin/env python3
"""
Build Installer for HR-TEM Analyzer

Creates platform-specific installers:
- Windows: .exe installer using Inno Setup or NSIS
- macOS: .dmg disk image
- Linux: .deb and .rpm packages (if fpm installed)

Requirements:
    pip install pyinstaller

    Windows: Install Inno Setup from https://jrsoftware.org/isinfo.php
    macOS: Uses built-in tools (hdiutil)
    Linux: Install fpm (gem install fpm)

Usage:
    python scripts/build_installer.py           # Build for current platform
    python scripts/build_installer.py --all     # Build all formats
    python scripts/build_installer.py --clean   # Clean before build
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


class InstallerBuilder:
    """Builds platform-specific installers"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dist_dir = project_root / 'dist'
        self.build_dir = project_root / 'build'
        self.installer_dir = project_root / 'installer'
        self.output_dir = project_root / 'installer_output'

        self.app_name = 'HRTEM-Analyzer'
        self.version = '1.0.0'

    def clean(self):
        """Clean build directories"""
        print("\n=== Cleaning build directories ===")
        for dir_path in [self.dist_dir, self.build_dir, self.output_dir]:
            if dir_path.exists():
                print(f"Removing {dir_path}...")
                shutil.rmtree(dir_path)

    def build_executable(self):
        """Build the executable using PyInstaller"""
        print("\n=== Building executable with PyInstaller ===")

        spec_file = self.project_root / 'hrtem_analyzer.spec'
        if spec_file.exists():
            cmd = [sys.executable, '-m', 'PyInstaller', '--noconfirm', str(spec_file)]
        else:
            cmd = [
                sys.executable, '-m', 'PyInstaller',
                '--noconfirm',
                '--windowed',
                '--name', self.app_name,
                str(self.project_root / 'scripts' / 'run_gui.py')
            ]

        subprocess.check_call(cmd, cwd=self.project_root)

        app_dir = self.dist_dir / self.app_name
        if not app_dir.exists():
            raise RuntimeError("PyInstaller build failed - no output directory")

        print(f"✓ Executable built: {app_dir}")

    def build_windows_installer(self):
        """Build Windows installer using Inno Setup"""
        print("\n=== Building Windows Installer ===")

        # Check for Inno Setup
        inno_paths = [
            r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
            r"C:\Program Files\Inno Setup 6\ISCC.exe",
            shutil.which("ISCC"),
        ]

        iscc = None
        for path in inno_paths:
            if path and Path(path).exists():
                iscc = path
                break

        iss_file = self.installer_dir / 'windows_installer.iss'

        if iscc and iss_file.exists():
            print("Using Inno Setup...")
            self.output_dir.mkdir(exist_ok=True)
            subprocess.check_call([iscc, str(iss_file)], cwd=self.project_root)
            print(f"✓ Windows installer created in {self.output_dir}")
        else:
            # Fallback: create self-extracting archive using 7-Zip or just ZIP
            print("Inno Setup not found. Creating ZIP archive instead...")
            self._create_zip_archive()

    def _create_zip_archive(self):
        """Create ZIP archive as fallback"""
        import zipfile

        self.output_dir.mkdir(exist_ok=True)
        zip_path = self.output_dir / f'{self.app_name}-{self.version}-Windows-portable.zip'

        app_dir = self.dist_dir / self.app_name

        print(f"Creating {zip_path}...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in app_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(app_dir.parent)
                    zf.write(file_path, arcname)

        print(f"✓ ZIP archive created: {zip_path}")
        print(f"  Size: {zip_path.stat().st_size / (1024*1024):.1f} MB")

    def build_macos_dmg(self):
        """Build macOS DMG installer"""
        print("\n=== Building macOS DMG ===")

        app_bundle = self.dist_dir / f'{self.app_name}.app'

        if not app_bundle.exists():
            # Check if it's in a folder instead
            app_bundle = self.dist_dir / self.app_name / f'{self.app_name}.app'

        if not app_bundle.exists():
            print("Creating app bundle from executable...")
            self._create_macos_app_bundle()
            app_bundle = self.dist_dir / f'{self.app_name}.app'

        self.output_dir.mkdir(exist_ok=True)
        dmg_path = self.output_dir / f'{self.app_name}-{self.version}-macOS.dmg'

        # Create DMG
        temp_dmg = self.output_dir / 'temp.dmg'

        # Create temporary DMG
        subprocess.check_call([
            'hdiutil', 'create',
            '-volname', self.app_name,
            '-srcfolder', str(app_bundle),
            '-ov', '-format', 'UDRW',
            str(temp_dmg)
        ])

        # Convert to compressed DMG
        subprocess.check_call([
            'hdiutil', 'convert',
            str(temp_dmg),
            '-format', 'UDZO',
            '-o', str(dmg_path)
        ])

        temp_dmg.unlink()
        print(f"✓ macOS DMG created: {dmg_path}")

    def _create_macos_app_bundle(self):
        """Create macOS app bundle structure"""
        app_dir = self.dist_dir / self.app_name
        bundle = self.dist_dir / f'{self.app_name}.app'

        # Create bundle structure
        contents = bundle / 'Contents'
        macos = contents / 'MacOS'
        resources = contents / 'Resources'

        macos.mkdir(parents=True)
        resources.mkdir(parents=True)

        # Copy executable
        for item in app_dir.iterdir():
            dest = macos / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        # Create Info.plist
        info_plist = contents / 'Info.plist'
        info_plist.write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>{self.app_name}</string>
    <key>CFBundleDisplayName</key>
    <string>HR-TEM Analyzer</string>
    <key>CFBundleIdentifier</key>
    <string>com.hrtem.analyzer</string>
    <key>CFBundleVersion</key>
    <string>{self.version}</string>
    <key>CFBundleShortVersionString</key>
    <string>{self.version}</string>
    <key>CFBundleExecutable</key>
    <string>{self.app_name}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeName</key>
            <string>TIFF Image</string>
            <key>CFBundleTypeExtensions</key>
            <array>
                <string>tif</string>
                <string>tiff</string>
            </array>
        </dict>
    </array>
</dict>
</plist>
''')

    def build_linux_packages(self):
        """Build Linux packages using fpm"""
        print("\n=== Building Linux Packages ===")

        if not shutil.which('fpm'):
            print("fpm not found. Creating tar.gz archive instead...")
            self._create_tar_archive()
            return

        app_dir = self.dist_dir / self.app_name
        self.output_dir.mkdir(exist_ok=True)

        # Build .deb package
        deb_path = self.output_dir / f'{self.app_name.lower()}_{self.version}_amd64.deb'
        subprocess.check_call([
            'fpm', '-s', 'dir', '-t', 'deb',
            '-n', self.app_name.lower(),
            '-v', self.version,
            '--description', 'HR-TEM Image Analyzer with Precision CD Measurement',
            '--license', 'MIT',
            '--prefix', f'/opt/{self.app_name}',
            '-p', str(deb_path),
            '-C', str(app_dir),
            '.'
        ])
        print(f"✓ DEB package created: {deb_path}")

        # Build .rpm package
        rpm_path = self.output_dir / f'{self.app_name.lower()}-{self.version}-1.x86_64.rpm'
        subprocess.check_call([
            'fpm', '-s', 'dir', '-t', 'rpm',
            '-n', self.app_name.lower(),
            '-v', self.version,
            '--description', 'HR-TEM Image Analyzer with Precision CD Measurement',
            '--license', 'MIT',
            '--prefix', f'/opt/{self.app_name}',
            '-p', str(rpm_path),
            '-C', str(app_dir),
            '.'
        ])
        print(f"✓ RPM package created: {rpm_path}")

    def _create_tar_archive(self):
        """Create tar.gz archive for Linux"""
        import tarfile

        self.output_dir.mkdir(exist_ok=True)
        tar_path = self.output_dir / f'{self.app_name}-{self.version}-Linux.tar.gz'
        app_dir = self.dist_dir / self.app_name

        print(f"Creating {tar_path}...")
        with tarfile.open(tar_path, 'w:gz') as tf:
            tf.add(app_dir, arcname=self.app_name)

        print(f"✓ TAR archive created: {tar_path}")
        print(f"  Size: {tar_path.stat().st_size / (1024*1024):.1f} MB")

    def build_for_current_platform(self):
        """Build for current platform"""
        self.build_executable()

        system = platform.system()
        if system == 'Windows':
            self.build_windows_installer()
        elif system == 'Darwin':
            self.build_macos_dmg()
        elif system == 'Linux':
            self.build_linux_packages()
        else:
            print(f"Unknown platform: {system}")
            self._create_zip_archive()

    def build_all(self):
        """Build all formats (only works on current platform)"""
        self.build_executable()

        system = platform.system()
        if system == 'Windows':
            self.build_windows_installer()
        elif system == 'Darwin':
            self.build_macos_dmg()
        elif system == 'Linux':
            self.build_linux_packages()

        # Always create a ZIP as well
        self._create_zip_archive()


def main():
    parser = argparse.ArgumentParser(description='Build HR-TEM Analyzer installer')
    parser.add_argument('--all', action='store_true', help='Build all formats')
    parser.add_argument('--clean', action='store_true', help='Clean before build')
    parser.add_argument('--exe-only', action='store_true', help='Build executable only (no installer)')

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    builder = InstallerBuilder(project_root)

    print("=" * 50)
    print("  HR-TEM Analyzer Installer Builder")
    print("=" * 50)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")

    if args.clean:
        builder.clean()

    if args.exe_only:
        builder.build_executable()
    elif args.all:
        builder.build_all()
    else:
        builder.build_for_current_platform()

    print("\n" + "=" * 50)
    print("  Build Complete!")
    print("=" * 50)
    print(f"\nOutput directory: {builder.output_dir}")

    if builder.output_dir.exists():
        print("\nCreated files:")
        for f in builder.output_dir.iterdir():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
