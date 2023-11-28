using Documenter
using SmoQyPT

makedocs(
    sitename = "SmoQyPT",
    format = Documenter.HTML(),
    modules = [SmoQyPT]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
