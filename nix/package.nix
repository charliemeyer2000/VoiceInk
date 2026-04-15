{
  lib,
  stdenvNoCC,
  fetchurl,
  _7zz,
  sourcesFile,
}:
let
  sourcesData = lib.importJSON sourcesFile;
  inherit (sourcesData) version;
in
stdenvNoCC.mkDerivation {
  pname = "voiceink";
  inherit version;

  src = fetchurl {
    inherit (sourcesData) url hash;
  };

  nativeBuildInputs = [_7zz];
  sourceRoot = ".";

  unpackPhase = ''
    7zz x "$src" -o"$sourceRoot" || true
  '';

  dontFixup = true;
  dontStrip = true;

  installPhase = ''
    runHook preInstall
    mkdir -p $out/Applications
    cp -r VoiceInk.app $out/Applications/
    runHook postInstall
  '';

  meta = {
    description = "Voice-to-text dictation app for macOS powered by whisper.cpp";
    homepage = "https://github.com/charliemeyer2000/VoiceInk";
    license = lib.licenses.gpl3Only;
    sourceProvenance = [lib.sourceTypes.binaryNativeCode];
    platforms = ["aarch64-darwin"];
  };
}
