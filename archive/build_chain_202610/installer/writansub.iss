; WritanSub installer (Inno Setup 6).
;
; Compile:
;   Full (offline bundle, ~3+ GB split into .exe + .bin disk-spanning parts):
;     ISCC.exe /DAppVersion=0.1.7.3 writansub.iss
;   Light (source bundle + install.bat, ~30 MB single file):
;     ISCC.exe /DAppVersion=0.1.7.3 /DLIGHT writansub.iss
;
; Expects build.ps1 (full) or light.ps1 (lite) to have produced the dist tree
; under ..\dist\WritanSub or ..\dist\WritanSub-light.

#define AppName     "WritanSub"
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppPublisher "WritanSub"
#define AppExeName   "WritanSub.exe"

#ifdef LIGHT
  #define SourceRoot         "..\dist\WritanSub-light"
  #define OutputBaseFileName "WritanSub-Setup-light-" + AppVersion
  #define InstallerSuffix    " (轻量版)"
#else
  #define SourceRoot         "..\dist\WritanSub"
  #define OutputBaseFileName "WritanSub-Setup-" + AppVersion
  #define InstallerSuffix    ""
#endif

[Setup]
AppId={{8E6F0E6E-4D3C-4F4D-9B6A-91F2A4B38E11}
AppName={#AppName}{#InstallerSuffix}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\{#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\release
OutputBaseFilename={#OutputBaseFileName}
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
#ifndef LIGHT
DiskSpanning=yes
DiskSliceSize=1900000000
#endif
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeName}
AllowNoIcons=yes

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"
Name: "startmenuicon"; Description: "创建开始菜单快捷方式"; GroupDescription: "附加任务:"

[Files]
#ifdef LIGHT
Source: "{#SourceRoot}\WritanSub.exe";          DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceRoot}\WritanSubCLI.exe";       DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceRoot}\install.bat";            DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceRoot}\README.txt";             DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "{#SourceRoot}\requirements.lock.txt";  DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceRoot}\app\*";                  DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceRoot}\vendor\*";               DestDir: "{app}\vendor"; Flags: ignoreversion recursesubdirs createallsubdirs
; runtime/ is intentionally NOT packaged — install.bat creates it from vendor/python-embed.zip
#else
Source: "{#SourceRoot}\WritanSub.exe";          DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceRoot}\WritanSubCLI.exe";       DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceRoot}\app\*";                  DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceRoot}\runtime\*";              DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif

[Icons]
Name: "{group}\{#AppName}";           Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: startmenuicon
Name: "{group}\卸载 {#AppName}";      Filename: "{uninstallexe}"; Tasks: startmenuicon
Name: "{userdesktop}\{#AppName}";     Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
#ifdef LIGHT
Filename: "{app}\install.bat";     Description: "立即下载依赖 (约 4 GB, 需要联网)"; Flags: postinstall shellexec waituntilterminated
Filename: "{app}\{#AppExeName}";   Description: "启动 {#AppName}"; Flags: postinstall shellexec nowait skipifsilent unchecked
#else
Filename: "{app}\{#AppExeName}";   Description: "启动 {#AppName}"; Flags: postinstall shellexec nowait skipifsilent
#endif

[UninstallDelete]
#ifdef LIGHT
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\vendor"
#endif
