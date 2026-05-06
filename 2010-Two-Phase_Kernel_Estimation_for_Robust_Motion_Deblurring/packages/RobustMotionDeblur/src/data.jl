# Data download and loading utilities for paper examples

const DEFAULT_DATA_URL = "https://www.cse.cuhk.edu.hk/~leojia/projects/robust_deblur/"

"""
    download_paper_data(target_dir="data"; base_url=DEFAULT_DATA_URL)

Best-effort download helper for the original paper's example images/kernels.
This assumes that the project page is still available and that files
are laid out in a simple structure.

**Note:** The paper website does not provide a direct file listing. Users
must manually download images from the project page and place them in
`target_dir`, or modify this function to add specific filenames.

To download images manually:
1. Visit: https://www.cse.cuhk.edu.hk/~leojia/projects/robust_deblur/
2. Download the desired images/kernels
3. Place them in the `target_dir` (default: "data")
"""
function download_paper_data(target_dir::AbstractString = "data"; base_url::AbstractString = DEFAULT_DATA_URL)
    mkpath(target_dir)
    # The exact filenames are not specified in the paper; users can adapt
    # this list based on the current project webpage.
    # Example: files = ["image1.jpg", "kernel1.mat", ...]
    files = String[]
    if isempty(files)
        @warn "No files specified for download. Please manually download images " *
              "from $base_url and place them in \"$target_dir\"."
        return target_dir
    end
    for rel in files
        url = base_url * rel
        dest = joinpath(target_dir, rel)
        mkpath(dirname(dest))
        try
            Downloads.download(url, dest)
        catch e
            @error "Failed to download $url: $e"
            rethrow()
        end
    end
    return target_dir
end
