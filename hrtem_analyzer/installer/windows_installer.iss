; HR-TEM Analyzer Installer Script (Inno Setup)
; =============================================
;
; Requirements:
;   1. Download Inno Setup from https://jrsoftware.org/isinfo.php
;   2. Build the executable first: python scripts/build_executable.py
;   3. Compile this script with Inno Setup Compiler
;
; Or use: python scripts/build_installer.py (automated)

#define MyAppName "HR-TEM Analyzer"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "HR-TEM Analyzer Team"
#define MyAppURL "https://github.com/example/hrtem-analyzer"
#define MyAppExeName "HRTEM-Analyzer.exe"

[Setup]
; Application information
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}

; Installation paths
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes

; Output settings
OutputDir=installer_output
OutputBaseFilename=HRTEM-Analyzer-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes

; Appearance
WizardStyle=modern
SetupIconFile=resources\icon.ico

; Privileges
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Misc
DisableWelcomePage=no
LicenseFile=LICENSE
InfoBeforeFile=README.md

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; Main application files (from PyInstaller output)
Source: "dist\HRTEM-Analyzer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Sample files (optional)
; Source: "samples\*"; DestDir: "{app}\samples"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Registry]
; File associations for .tif and .tiff files (optional)
Root: HKCU; Subkey: "Software\Classes\.tif\OpenWithProgids"; ValueType: string; ValueName: "HRTEMAnalyzer.tif"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.tiff\OpenWithProgids"; ValueType: string; ValueName: "HRTEMAnalyzer.tiff"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\HRTEMAnalyzer.tif"; ValueType: string; ValueName: ""; ValueData: "HR-TEM Image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\HRTEMAnalyzer.tif\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

[Code]
// Check if .NET or VC++ redistributable is needed (add if required)
function InitializeSetup: Boolean;
begin
  Result := True;
  // Add any prerequisite checks here
end;
