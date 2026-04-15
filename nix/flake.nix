{
  description = "VoiceInk — macOS dictation app (pre-built from fork)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {
    self,
    nixpkgs,
  }: let
    systems = ["aarch64-darwin"];
    forAllSystems = nixpkgs.lib.genAttrs systems;

    versionFiles = builtins.readDir ./versions;
    versionNames =
      builtins.map (f: nixpkgs.lib.removeSuffix ".json" f)
      (builtins.filter (f: nixpkgs.lib.hasSuffix ".json" f)
        (builtins.attrNames versionFiles));
    latestVersion =
      builtins.head
      (builtins.sort (a: b: builtins.compareVersions a b > 0) versionNames);
  in {
    packages = forAllSystems (system: let
      pkgs = import nixpkgs {inherit system;};
      mkVoiceInk = sourcesFile:
        pkgs.callPackage ./package.nix {inherit sourcesFile;};
      latestSourcesFile = ./versions/${latestVersion + ".json"};
    in {
      voiceink = mkVoiceInk latestSourcesFile;
      default = self.packages.${system}.voiceink;
    });

    overlays.default = _final: prev: {
      voiceink = self.packages.${prev.stdenv.hostPlatform.system}.voiceink;
    };
  };
}
