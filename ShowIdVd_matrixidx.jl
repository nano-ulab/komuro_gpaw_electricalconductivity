# Initial setting---------------------------------------

using CairoMakie
using CSVFiles, FileIO, DataFrames

include(joinpath(joinpath(splitpath(@__DIR__)[1:end-1]), "Function", "DataTrimmer.jl"))
include(joinpath(joinpath(splitpath(@__DIR__)[1:end-1]), "Function", "FileManager.jl"))


# User Control parameters-------------------------------

Search_scope			= ["DFTScripts/cif_files"]
IdVd_data_ext			= [".csv"]
xlabel = L"\textbf{V_d~\textrm{(V)}}"
ylabel = L"\textbf{I_d~\textrm{(A)}}"
savename = "/home/lune/Pictures/opt_cif_IdVd.png"

# Collect data------------------------------------------

function show_description(
		Search_scope::Union{Nothing, Vector{String}},
		IdVd_data_ext::Union{Nothing, Vector{String}},
		xlabel,
		ylabel
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

	rows = 2
    columns = 4
	axsize = (300, 300)
	framesize = (2/3).*axsize
	figsize = (axsize[2]*columns, axsize[1]*rows)

	fig			= Figure(size = figsize)
	ax_matrix = Matrix{Axis}(undef, rows, columns)

    for r in 1:rows
        for c in 1:columns
            ax_matrix[r, c] = Axis(fig[r, c], title = filenames_IdVd[(r-1)*columns+c], yscale = log10, titlesize = 9.0f0,		xlabel=xlabel, ylabel=ylabel, xlabelsize=15.0f0, ylabelsize=15.0f0, xticklabelsize=15.0f0, yticklabelsize=15.0f0, xgridvisible=false, ygridvisible=false,		xticksmirrored=true, xminorticksvisible=true,		yticksmirrored=true, yminorticksvisible=true,		xtickalign=1, xminortickalign=1, ytickalign=1, yminortickalign=1, xticksize = 8, xminorticksize = 4, yticksize = 8, yminorticksize = 4, xticklabelpad = 3.0, yticklabelpad = 3.0)
        end
    end
	
	for r in 1:rows
		rowsize!(fig.layout, r, Fixed(framesize[2]))
	end
	
	for c in 1:columns
		colsize!(fig.layout, c, Fixed(framesize[1]))
	end

    for r in 1:rows
        for c in 1:columns
			ylims!(ax_matrix[r, c], 1e-11, 1e-6)
        end
    end
	
	num_data_IdVd	= 0
	
	
	if filepaths_IdVd != []
		num_data_IdVd	= length(filenames_IdVd)
	end
	
	colors = 1:num_data_IdVd

	for idx in 1:num_data_IdVd
		lines!(ax_matrix[(idx-1)÷columns+1, (idx-1)%columns+1], results_IdVd[idx][1], results_IdVd[idx][2]; label = filenames_IdVd[idx], color = colors[idx], colormap = :viridis, colorrange = (1, length(colors)))
	end
	
	display(fig)
	return fig	
end

figure = show_description(
	Search_scope,
	IdVd_data_ext,
	xlabel,
	ylabel
)

!isnothing(savename) ? save(savename, figure) : nothing
