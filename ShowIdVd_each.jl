# Initial setting---------------------------------------

using CairoMakie
using CSVFiles, FileIO, DataFrames

include(joinpath(joinpath(splitpath(@__DIR__)[1:end-1]), "Function", "DataTrimmer.jl"))
include(joinpath(joinpath(splitpath(@__DIR__)[1:end-1]), "Function", "FileManager.jl"))


# User Control parameters-------------------------------

Search_scope			= ["DFTScripts/csv_files_opt"]
IdVd_data_ext			= [".csv"]
xlabel = L"\textbf{V_d~\textrm{(V)}}"
ylabel = L"\textbf{I_d~\textrm{(A)}}"
savename = nothing
savename = "/home/lune/Pictures/opt_cif_IdVd"
saveformat = ".svg"

# Collect data------------------------------------------

function show_description(
		Search_scope::Union{Nothing, Vector{String}},
		IdVd_data_ext::Union{Nothing, Vector{String}},
		xlabel,
		ylabel,
		savename,
		saveformat
	)

	path_data		= joinpath(joinpath(splitpath(@__DIR__)[1:end-1]), joinpath(Search_scope))
	
	filepaths_IdVd	= (filecollector(path_data, detect_exts = IdVd_data_ext))

	@show filepaths_IdVd
	
	println("### filenames_IdVd ###")
	if filepaths_IdVd != []
		@show filenames_IdVd	= filepaths_IdVd	.|> basename .|> x -> splitext(x)[1]
		@show results_IdVd	= filepaths_IdVd	.|> x -> wavedatapacker(x; ignore_icon = "#")
		results_IdVd = results_IdVd .|> x -> [x[1], abs.(x[2])]
	end
	println()
	
	
	# Draw figure-------------------------------------------

	num_data_IdVd	= 0
	
	if filepaths_IdVd != []
		num_data_IdVd	= length(filenames_IdVd)
	end

	colors = 1:num_data_IdVd

	for (i, result) in enumerate(results_IdVd) 
		fig	= Figure()

        ax= Axis(fig[1, 1], yscale = log10, titlesize = 20.0f0,		xlabel=xlabel, ylabel=ylabel, xlabelsize=30.0f0, ylabelsize=30.0f0, xticklabelsize=30.0f0, yticklabelsize=30.0f0, xgridvisible=false, ygridvisible=false,		xticksmirrored=true, xminorticksvisible=true,		yticksmirrored=true, yminorticksvisible=true,		xtickalign=1, xminortickalign=1, ytickalign=1, yminortickalign=1, xticksize = 8, xminorticksize = 4, yticksize = 8, yminorticksize = 4, xticklabelpad = 3.0, yticklabelpad = 3.0)	
	
		ylims!(ax, 1e-11, 1e-6)

		lines!(ax, result[1], result[2]; label = filenames_IdVd[i], color = colors[i], colormap = :viridis, colorrange = (1, length(colors)))
	
		display(fig)
 
		if !isnothing(savename)
			if saveformat == ".png" || saveformat == ".svg" || saveformat == ".jpg"
			save(savename*"_"*filenames_IdVd[i]*saveformat, fig)
			else
				throw(ErrorException("!!! ERROR : saveformat must be chosen from .png, .svg, .jpg !!!")) 
			end
		end
	end
end

show_description(
	Search_scope,
	IdVd_data_ext,
	xlabel,
	ylabel,
	savename,
	saveformat
)
