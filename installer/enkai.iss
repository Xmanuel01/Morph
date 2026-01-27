; Enkai Windows installer (Inno Setup)
; Build with: ISCC /DMyAppVersion=<version> installer\enkai.iss

#define MyAppName "Enkai"
#define MyAppVersion "0.0.0"
#define MyAppPublisher "Emmanuel Odhiambo Onyango"
#define MyAppURL "https://github.com/Xmanuel01/Enkai"
#define MyAppExeName "enkai.exe"

[Setup]
AppId={{3A301351-9376-4C6F-9E7B-4BA9C5A54D0D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={userpf}\Enkai
DefaultGroupName=Enkai
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=enkai-setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes

[Files]
Source: "target\release\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Enkai"; Filename: "{app}\{#MyAppExeName}"

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Flags: preservestringtype; Check: NeedsAddPath

[Code]
function NeedsAddPath(): Boolean;
begin
  Result := Pos(Lowercase(ExpandConstant('{app}')), Lowercase(GetEnv('Path'))) = 0;
end;
