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
      mkJupyterSupport =
        pkgs:
        let
          kernelName = "llm-papers-replicas";
          pythonEnv = pkgs.python313.withPackages (
            ps: with ps; [
              huggingface-hub
              ipykernel
              jupyterlab
            ]
          );
          ensureJupyterKernel = pkgs.writeShellScriptBin "ensure-jupyter-kernel" ''
            set -euo pipefail

            kernel_name=${lib.escapeShellArg kernelName}
            export kernel_name

            ${pythonEnv}/bin/python -m ipykernel install \
              --user \
              --name "$kernel_name" \
              --display-name "$kernel_name"
          '';
          startJupyter = pkgs.writeShellScriptBin "start-jupyter" ''
            set -euo pipefail

            kernel_name=${lib.escapeShellArg kernelName}
            export kernel_name

            ${ensureJupyterKernel}/bin/ensure-jupyter-kernel >/dev/null

            exec ${pythonEnv}/bin/jupyter-lab \
              --no-browser \
              --ip="*" \
              --NotebookApp.token="" \
              --NotebookApp.password="" \
              --ServerApp.disable_check_xsrf=True \
              "$@"
          '';
        in
        {
          inherit
            kernelName
            pythonEnv
            ensureJupyterKernel
            startJupyter
            ;
        };
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
          jupyter = mkJupyterSupport pkgs;
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
                # for GGUF / encoder weights) and Jupyter tooling
                jupyter.pythonEnv
                jupyter.ensureJupyterKernel
                jupyter.startJupyter
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
              export kernel_name=${lib.escapeShellArg jupyter.kernelName}
              ensure-jupyter-kernel >/dev/null
              echo
            '';
          };
        }
      );

      checks = forAllSystems (
        system: pkgs:
        let
          jupyter = mkJupyterSupport pkgs;
        in
        {
          enterTest = pkgs.runCommand "openmythos-enter-test" { nativeBuildInputs = [ pkgs.git ]; } ''
            git --version | grep -F "${pkgs.git.version}" >/dev/null

            if command -v llama >/dev/null 2>&1 || command -v llama-cli >/dev/null 2>&1; then
              echo "llama.cpp available" > "$out"
            else
              echo "Note: llama.cpp not on PATH yet (CPU-only package remains disabled, matching devenv.nix)" > "$out"
            fi
          '';
          jupyterKernelTest = pkgs.runCommand "openmythos-jupyter-kernel-test" {
            nativeBuildInputs = [
              jupyter.ensureJupyterKernel
              jupyter.pythonEnv
              jupyter.startJupyter
            ];
          } ''
            export HOME="$(mktemp -d)"
            export XDG_DATA_HOME="$HOME/.local/share"
            export kernel_name=${lib.escapeShellArg jupyter.kernelName}

            ensure-jupyter-kernel
            [ -f "$XDG_DATA_HOME/jupyter/kernels/$kernel_name/kernel.json" ]
            start-jupyter --help >/dev/null

            echo "kernel $kernel_name ready on ${system}" > "$out"
          '';
        }
      );

      formatter = forAllSystems (_system: pkgs: pkgs.nixfmt-rfc-style);
    };
}
