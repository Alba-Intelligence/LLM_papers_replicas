{
  description = "OpenMythos development environment";

  inputs = {
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";
    llms.url = "github:numtide/llm-agents.nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      llms,
      ...
    }:
    let
      lib = nixpkgs.lib;
      systems = builtins.attrNames llms.packages;
      forAllSystems =
        f:
        lib.genAttrs systems (
          system:
          f system (
            import nixpkgs {
              inherit system;
              config = {
                allowUnfree = true;
                cudaSupport = false;
              };
            }
          )
        );
    in
    {
      devShells = forAllSystems (
        system: pkgs:
        let
          llmsPkgs = llms.packages.${system};
          # CPU-only llama.cpp (no CUDA) for dev environments; kept disabled to
          # match the current devenv.nix package list.
          llamaCppCpu = pkgs.llama-cpp.override { cudaSupport = false; };
        in
        {
          default = pkgs.mkShell {
            packages =
              (with pkgs; [
                # utils
                gh
                git

                # IDE
                bash
                vscode

                # Language runtime
                julia-bin

                # LLM - CPU-only llama.cpp (CLI + server); workflow uses
                # llama-cpp instead of ollama.
                # llamaCppCpu

                # Python - Hugging Face Hub CLI (e.g. huggingface-cli download
                # for GGUF / encoder weights)
                python313Packages.huggingface-hub
              ])
              ++ (with llmsPkgs; [
                claude-code
                copilot-cli
                kilocode-cli
                opencode
                openskills
                openspec
                pi
                spec-kit
              ]);

            shellHook = ''
              echo
            '';
          };
        }
      );

      checks = forAllSystems (
        _system: pkgs: {
          enterTest = pkgs.runCommand "openmythos-enter-test" { nativeBuildInputs = [ pkgs.git ]; } ''
            git --version | grep -F "${pkgs.git.version}" >/dev/null

            if command -v llama >/dev/null 2>&1 || command -v llama-cli >/dev/null 2>&1; then
              echo "llama.cpp available" > "$out"
            else
              echo "Note: llama.cpp not on PATH yet (CPU-only package remains disabled, matching devenv.nix)" > "$out"
            fi
          '';
        }
      );

      formatter = forAllSystems (_system: pkgs: pkgs.nixfmt-rfc-style);
    };
}
