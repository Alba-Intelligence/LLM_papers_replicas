#!/usr/bin/env julia

using RobustMotionDeblur

# Best-effort download of paper data; user may need to adjust URLs.

dir = download_paper_data()
println("Data downloaded to: ", dir)
