"""
Test to verify AGENTS.md exists and contains required standards.
"""
using Test

@testset "AGENTS.md verification" begin
    agents_file = joinpath(@__DIR__, "..", "AGENTS.md")
    
    @testset "AGENTS.md exists" begin
        @test isfile(agents_file)
    end
    
    @testset "AGENTS.md contains at least 5 distinct standards" begin
        content = read(agents_file, String)
        
        # Count distinct standard sections (## headings)
        sections = [m.match for m in eachmatch(r"^##\s+[^\n]+", content, multiline=true)]
        
        # Required sections (at least 5):
        required_keywords = [
            "Package Structure",
            "Coding Standards",
            "TDD",
            "CLI",
            "Documentation",
            "Dependency"
        ]
        
        found_sections = 0
        for keyword in required_keywords
            if occursin(keyword, content)
                found_sections += 1
            end
        end
        
        @test found_sections >= 5 "AGENTS.md should contain at least 5 distinct development standards, found $found_sections"
    end
end
