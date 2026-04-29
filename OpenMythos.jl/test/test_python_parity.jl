using Test

function _python_parity_runner()
    if haskey(ENV, "OPENMYTHOS_PARITY_PYTHON")
        return Cmd([ENV["OPENMYTHOS_PARITY_PYTHON"]])
    end
    return Cmd(["uv", "run", "--with", "torch", "python"])
end

function _run_python_parity(args::Vector{String})
    cmd = `$(_python_parity_runner()) test/python_reference_smoke.py $(args...)`
    return read(cmd, String)
end

function _parse_float_line(line::AbstractString)
    isempty(line) && return Float32[]
    return parse.(Float32, split(line, ','))
end

@testset "Python reference parity" begin
    if get(ENV, "OPENMYTHOS_ENABLE_PYTHON_PARITY", "0") != "1"
        @test true
    else
        dim = 8
        max_len = 6
        theta = 500000.0f0

        py = split(chomp(_run_python_parity(["precompute_rope", string(dim), string(max_len), string(theta)])), '\n')
        @test length(py) == 3
        @test py[1] == "$max_len $(dim ÷ 2)"

        julia_freqs = precompute_rope_freqs(dim, max_len; theta=theta)
        py_real = reshape(_parse_float_line(py[2]), max_len, dim ÷ 2)
        py_imag = reshape(_parse_float_line(py[3]), max_len, dim ÷ 2)
        @test all(isapprox.(Float32.(real.(julia_freqs)), py_real; atol=1f-6))
        @test all(isapprox.(Float32.(imag.(julia_freqs)), py_imag; atol=1f-6))

        b, t, h, d = 1, 4, 2, 8
        py_rope = split(chomp(_run_python_parity(["apply_rope", string(b), string(t), string(h), string(d), string(theta)])), '\n')
        @test py_rope[1] == "$b $t $h $d"
        py_vals = reshape(_parse_float_line(py_rope[2]), b, t, h, d)

        x = reshape(Float32.(1:(b * t * h * d)) ./ 100f0, b, t, h, d)
        julia_out = apply_rope(x, precompute_rope_freqs(d, t; theta=theta))
        @test all(isapprox.(julia_out, py_vals; atol=1f-5))
    end
end
