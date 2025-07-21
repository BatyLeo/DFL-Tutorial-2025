### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ e36da6f2-b236-4b67-983c-152a7ff54e05
begin
    using Colors
    using DifferentiableFrankWolfe
    using Flux
    using FrankWolfe
    using Graphs
    using GridGraphs
    using Images
    using InferOpt
    using LaTeXStrings
    using LinearAlgebra
    using Markdown: MD, Admonition, Code
    using Metalhead
    using NPZ
    using Plots
    using ProgressLogging
    using Random
    using PlutoTeachingTools
    using PlutoUI
    using UnicodePlots
    using Zygote
    Random.seed!(63)
end;

# ╔═╡ 8b7876e4-2f28-42f8-87a1-459b665cff30
md"""
Imports all package dependencies (this may take a while to run the first time)
"""

# ╔═╡ e279878d-9c8d-47c8-9453-3aee1118818b
md"""
**Utilities (hidden)**
"""

# ╔═╡ b5b0bb58-9e02-4551-a9ba-0ba0ffceb350
TableOfContents(depth=3)

# ╔═╡ b0616d13-41fa-4a89-adb3-bf8b27b13657
info(text; title="Info") = MD(Admonition("info", title, [text]));

# ╔═╡ 9adcecda-eaeb-4432-8634-a1ce868c50f5
logocolors = Colors.JULIA_LOGO_COLORS;

# ╔═╡ 21bee304-8aab-4c57-b3ab-ceec6a608320
function get_angle(v)
    @assert !(norm(v) ≈ 0)
    v = v ./ norm(v)
    if v[2] >= 0
        return acos(v[1])
    else
        return π + acos(-v[1])
    end
end;

# ╔═╡ 2067c125-f473-4cc2-a548-87b1b0ad9011
function init_plot(title)
    pl = plot(;
        aspect_ratio=:equal,
        legend=:outerleft,
        xlim=(-1.1, 1.1),
        ylim=(-1.1, 1.1),
        title=title,
    )
    return pl
end;

# ╔═╡ d85d1e30-92f4-4bc7-8da9-3b417f51530b
function plot_polytope!(pl, vertices)
    plot!(
        vcat(map(first, vertices), first(vertices[1])),
        vcat(map(last, vertices), last(vertices[1]));
        fillrange=0,
        fillcolor=:gray,
        fillalpha=0.2,
        linecolor=:black,
        label=L"\mathrm{conv}(\mathcal{V})"
    )
end;

# ╔═╡ 7e46ec11-b0ff-4dc7-9939-32ad154aeb96
function plot_objective!(pl, θ)
    plot!(
        pl,
        [0., θ[1]],
        [0., θ[2]],
        color=logocolors.purple,
        arrow=true,
        lw=2,
        label=nothing
    )
    Plots.annotate!(
        pl,
        [-0.2 * θ[1]],
        [-0.2 * θ[2]],
        [L"\theta"],
    )
    return pl
end;

# ╔═╡ 95a43871-924b-4ff1-87ac-76c33d22c9ad
function plot_maximizer!(pl, θ, polytope, maximizer)
    ŷ = maximizer(θ; polytope)
    scatter!(
        pl,
        [ŷ[1]],
        [ŷ[2]];
        color=logocolors.red,
        markersize=9,
        markershape=:square,
        label=L"f(\theta)"
    )
end;

# ╔═╡ 269547da-f4ec-4746-9453-5cb8d7703da8
function plot_distribution!(pl, probadist)
    A = probadist.atoms
    As = sort(A, by=get_angle)
    p = probadist.weights
    plot!(
        pl,
        vcat(map(first, As), first(As[1])),
        vcat(map(last, As), last(As[1]));
        fillrange=0,
        fillcolor=:blue,
        fillalpha=0.1,
        linestyle=:dash,
        linecolor=logocolors.blue,
        label=L"\mathrm{conv}(\hat{p}(\theta))"
    )
    scatter!(
        pl,
        map(first, A),
        map(last, A);
        markersize=25 .* p .^ 0.5,
        markercolor=logocolors.blue,
        markerstrokewidth=0,
        markeralpha=0.4,
        label=L"\hat{p}(\theta)"
    )
end;

# ╔═╡ 68c6b115-5873-4678-9f3a-54b72554e8d3
function plot_expectation!(pl, probadist)
    ŷΩ = compute_expectation(probadist)
    scatter!(
        pl,
        [ŷΩ[1]],
        [ŷΩ[2]];
        color=logocolors.blue,
        markersize=6,
        markershape=:hexagon,
        label=L"\hat{f}(\theta)"
    )
end;

# ╔═╡ d9dbc402-383a-4aad-9f44-08f06b41ab0d
function compress_distribution!(
    probadist::FixedAtomsProbabilityDistribution{A,W}; atol=0
) where {A,W}
    (; atoms, weights) = probadist
    to_delete = Int[]
    for i in length(probadist):-1:1
        ai = atoms[i]
        for j in 1:(i-1)
            aj = atoms[j]
            if isapprox(ai, aj; atol=atol)
                weights[j] += weights[i]
                push!(to_delete, i)
                break
            end
        end
    end
    sort!(to_delete)
    deleteat!(atoms, to_delete)
    deleteat!(weights, to_delete)
    return probadist
end;

# ╔═╡ 78312b73-42bd-42d3-b31d-83222fd8fbaa
set_angle_oracle = md"""
angle = $(@bind angle_oracle Slider(0:0.01:2π; default=π, show_value=false))
""";

# ╔═╡ 4678209f-9bb9-4d3b-b031-575f2fba4916
set_angle_perturbed = md"""
angle = $(@bind angle_perturbed Slider(0:0.01:2π; default=π, show_value=false))
""";

# ╔═╡ 3bb99c85-35de-487d-a5e7-1cd1313fd6ea
set_nb_samples_perturbed = md"""
samples = $(@bind nb_samples_perturbed Slider(1:500; default=10, show_value=true))
""";

# ╔═╡ d447f8af-78de-4306-ba24-22851c366690
set_epsilon_perturbed = md"""
epsilon = $(@bind epsilon_perturbed Slider(0.0:0.02:1.0; default=0.0, show_value=true))
""";

# ╔═╡ f5afa452-9485-4dba-93fe-277d87ad0344
set_plot_probadist_perturbed = md"""
Plot probability distribution? $(@bind plot_probadist_perturbed CheckBox())
""";

# ╔═╡ 14af0338-554a-4f71-a290-3b4f16cc6af5
md"""
# Shortest paths on warcraft maps
"""

# ╔═╡ 2901d761-405a-4800-b1a7-d2a80cf8aea5
ChooseDisplayMode()

# ╔═╡ 23f7f158-9a74-4f6b-9718-5609f458b101
md"""
Assignment:
- Due next wednesday
- Send by email a written pdf report, as well as your final notebook/code
- Each green question box expects a written answer.
- TODO boxes expect some code implementation, and eventually some comments and analyis in the report.
"""

# ╔═╡ 978b5cff-bd07-48a1-8248-366798bf5d35
tip(md"""This file is a [Pluto](https://plutojl.org/) notebook. There are some differences respect to Jupyter notebooks you may be familiar with:
- It's a regular julia code file.
- **Self-contained** environment: packages are managed and installed directly in each notebook.
- **Reactivity** and interactivity: cells are connected, such that when you modify a variable value, all other cells depending on it (i.e. using this variable) are automatically reloaded and their outputs updated. Feel free to modify some variables to observe the effects on the other cells. This allow interactivity with tools such as dropdown and sliders.
- Some cells are hidden by default, if you want to see their content, just click on the eye icon on its top left.

If you prefer, you can transfer the code into a regular julia script for the trainings at the end, it might be more practical for long running times.
""")

# ╔═╡ f13bf21c-33db-4620-add8-bfd82f493d7c
md"""
# 1. Recap on CO-ML pipelines
"""

# ╔═╡ c8bf8d9a-783c-41c6-bf33-dc423e249d0b
tip(md"In this first section, feel free to play around with the parameters, and report your findings in the pdf report.")

# ╔═╡ f99d6992-dc3e-41d1-8922-4958886dade2
md"""

**Points of view**: 
1. Enrich learning pipelines with combinatorial algorithms.
2. Enhance combinatorial algorithms with learning pipelines.

```math
\xrightarrow[x]{\text{Instance}}
\fbox{ML predictor}
\xrightarrow[\theta]{\text{Objective}}
\fbox{CO algorithm}
\xrightarrow[y]{\text{Solution}}
```

**Challenge:** Differentiating through CO algorithms.

**Two main learning settings:**
- Learning by imitation: instances with labeled solutions $(x_i, y_i)_i$.
- Learning by experience: no labeled solutions $(x_i)_i$.
"""

# ╔═╡ 0d20da65-1e53-4b6e-b302-28243c94fb4c
md"""
## Many possible applications in both fields

- Shortest paths on Warcraft maps
- Stochastic Vehicle Scheduling
- Two-stage Minimum Spanning Tree
- Single-machine scheduling
- Dynamic Vehicle Routing
- ...
"""

# ╔═╡ 87040fd6-bd1a-47dd-875c-2caf5b50d2ce
md"""
## Smoothing by regularization

```math
\xrightarrow[\text{instance $x$}]{\text{Problem}}
\fbox{NN $\varphi_w$}
\xrightarrow[\text{direction $\theta$}]{\text{Objective}}
\fbox{MILP $\underset{y \in \mathcal{Y}}{\mathrm{argmax}} ~ \theta^\top y$}
\xrightarrow[\text{solution $\widehat{y}$}]{\text{Candidate}}
```

The combinatorial layer function

```math
f\colon \theta\longmapsto \underset{y \in \mathcal{Y}}{\mathrm{argmax}} ~ \theta^\top y
```
is piecewise constant $\implies$ no gradient information.

The perturbed regularized optimizer is defined by:

```math
\hat{f}_\varepsilon(\theta) = \mathbb{E}_{Z}\big[ \underset{y \in \mathcal{Y}}{\mathrm{argmax}} (\theta + \varepsilon Z)^\top y \big]
```
with ``Z\sim\mathcal{N}(0, 1)``, ``\varepsilon>0``.

``\implies`` becomes differentiable.

Can be seen as an expectation over the vertices of $\mathrm{conv}(\mathcal{Y})$.

```math

\hat{f}_\varepsilon(\theta) = \mathbb{E}_{\hat{p}(\cdot|\theta)}[Y] = \sum_{y\in\mathcal{Y}}~y~\hat{p}(y|\theta)
```
"""

# ╔═╡ 53f7468d-0015-4339-8e27-48812f541329
md"""
## Linear oracle
"""

# ╔═╡ 81cd64c4-f317-4555-ab7a-9a5b2b837b91
md"""Let's build a polytope with `N` vertices, and visualize perturbations and loss over it."""

# ╔═╡ 95013865-885c-4e0d-a76b-c2452b10bdad
N = 7

# ╔═╡ afdd0ea0-054a-4b8e-a7ea-8a21d1e021ff
polytope = [[cospi(2k / N), sinpi(2k / N)] for k in 0:N-1];

# ╔═╡ 61c624df-384a-4f01-a1c2-20d09d43aa74
md"""Combinatorial oracle: ``f(\theta; x) = \arg\max_{y\in\mathcal{Y}(x)} \theta^\top y``"""

# ╔═╡ 4b59b997-dc7d-49a9-8557-87d908673c22
maximizer(θ; polytope) = polytope[argmax(dot(θ, v) for v in polytope)];

# ╔═╡ 0fe3676f-70a4-4730-9ce9-ac5bc4204284
md"""
Here is a figure of the polytope and the armax output of the oracle in red.

You can modify θ by using the slider below to modify its angle:
"""

# ╔═╡ 446cb749-c1ec-46a1-8cff-74a99d0cc2d9
let
    θ = 0.5 .* [cos(angle_oracle), sin(angle_oracle)]
    pl = init_plot("Linear oracle")
    plot_polytope!(pl, polytope)
    plot_objective!(pl, θ)
    plot_maximizer!(pl, θ, polytope, maximizer)
    pl
end

# ╔═╡ 7db83f4b-0f9c-4d27-9a5e-bc6aacdae186
set_angle_oracle

# ╔═╡ 1ca1d8a3-d7bc-4386-8142-29c5cf2a87a0
md"""We use the [`Zygote.jl`](https://fluxml.ai/Zygote.jl/stable/) automatic differentiation library to compute the jacobian of our CO oracle with respect to ``\theta``.
"""

# ╔═╡ f370c7c5-0f39-4efa-a298-d913a591412d
let
    θ = 0.5 .* [cos(angle_oracle), sin(angle_oracle)]
    jac = Zygote.jacobian(θ -> maximizer(θ; polytope), θ)[1]
    @info "" θ = θ jacobian = jac
end

# ╔═╡ 2773083a-0f6a-4c28-8da3-2a4ee4efdb6f
question_box(md"1. Why is the jacobian zero for all values of ``\theta``?")

# ╔═╡ e6efe06c-8833-4a6b-8086-b7ebe91ee703
md"""## Perturbed Layer"""

# ╔═╡ 381c10e9-25c2-4ec0-8c37-48f7099abd03
md"""[`InferOpt.jl`](https://github.com/axelparmentier/InferOpt.jl) provides the `PerturbedAdditive` wrapper to regularize any given combinatorial optimization oracle $f$, and transform it into $\hat f$.

It takes the maximizer as the main arguments, and several optional keyword arguments such as:
- `ε`: size of the perturbation (=1 by default)
- `nb_samples`: number of Monte Carlo samples to draw for estimating expectations (=1 by default)

See the [documentation](https://axelparmentier.github.io/InferOpt.jl/dev/) for more details.
"""

# ╔═╡ 0a205ed0-e52d-4017-b78f-23c7447063f3
perturbed_layer = PerturbedAdditive(
    maximizer;
    ε=epsilon_perturbed,
    nb_samples=nb_samples_perturbed,
    seed=0
)

# ╔═╡ ded783ba-7b43-4351-8951-a452e1e26e3c
md"""Now we can visualize the perturbed maximizer output"""

# ╔═╡ 98c6fffd-26d2-4d94-ba99-1f3a59197079
TwoColumn(set_angle_perturbed, set_epsilon_perturbed)

# ╔═╡ b3fd69fb-a1dd-4102-9ced-eb0566821a57
TwoColumn(set_nb_samples_perturbed, set_plot_probadist_perturbed)

# ╔═╡ cc978152-4e1a-4aa1-9dc2-7e49f03ead76
let
    θ = 0.5 .* [cos(angle_perturbed), sin(angle_perturbed)]
    probadist = compute_probability_distribution(
        perturbed_layer, θ; polytope,
    )
    compress_distribution!(probadist)
    pl = init_plot("Perturbation")
    plot_polytope!(pl, polytope)
    plot_objective!(pl, θ)
    plot_probadist_perturbed && plot_distribution!(pl, probadist)
    plot_maximizer!(pl, θ, polytope, maximizer)
    plot_expectation!(pl, probadist)
    pl
end

# ╔═╡ f2926b6a-1ff0-4157-a6b8-a56f658f4d49
md"""When $\varepsilon > 0$, the perturbed maximizer is differentiable:"""

# ╔═╡ 712c87ea-91e0-4eaa-807c-6876ee5b311f
let
    θ = 0.5 .* [cos(angle_perturbed), sin(angle_perturbed)]
    Zygote.jacobian(θ -> perturbed_layer(θ; polytope), θ)[1]
end

# ╔═╡ 5dd28e66-afd8-4c9d-bc88-b87e5e13f390
question_box(md"2. What can you say about the derivatives of the perturbed maximizer?")

# ╔═╡ 6801811b-f68a-43b4-8b78-2f27c0dc6331
md"""
## Fenchel-Young loss (learning by imitation)
By defining:

```math
F^+_\varepsilon (\theta) := \mathbb{E}_{Z}\big[ \operatorname{max}_{y \in \mathcal{Y}(x)} (\theta + \varepsilon Z)^\top y \big],
```
and ``\Omega_\varepsilon^+`` its Fenchel conjugate, we can define the Fenchel-Young loss as follows:
```math
\mathcal{L}_{\varepsilon}^{\text{FY}}(\theta, \bar{y}) = F^+_\varepsilon (\theta) + \Omega_\varepsilon(\bar{y}) - \theta^\top \bar{y}
```

Given a target solution $\bar{y}$ and a parameter $\theta$, a subgradient is given by:
```math
\widehat{f}(\theta) - \bar{y} \in \partial_\theta \mathcal{L}_{\varepsilon}^{\text{FY}}(\theta, \bar{y}).
```
The optimization block has meaningful gradients $\implies$ we can backpropagate through the whole pipeline, using automatic differentiation.
"""

# ╔═╡ b748c794-b9b6-4e96-8f65-f34abd6b127e
question_box(md"3. What are the properties of ``\mathcal{L}_{\varepsilon}^{\text{FY}}?``")

# ╔═╡ 701f4d68-0424-4f9f-b904-84b52f6a4745
md"""Let's define the Fenchel-Young loss by using the `FenchelYoungLoss` wrapper from `InferOpt`:"""

# ╔═╡ d64790a7-6a02-44ca-a44f-268fea657690
fyl = FenchelYoungLoss(perturbed_layer)

# ╔═╡ 87c4d949-c7d9-4f70-8fe0-f273ad655635
md"""Let's visualize a contour plot of the loss with target ȳ fixed."""

# ╔═╡ e4bf4523-94c2-457f-9dd3-74f100d2dc17
X, Y = range(-1, 1, length=100), range(-1, 1, length=100);

# ╔═╡ e8ce60c1-4981-464a-9a9b-8ac5734a5bb4
TwoColumn(md"""Change `y_index` value to change target vertex ȳ:""", md"y\_index = $(@bind y_index Select(1:N))")

# ╔═╡ 6b212032-22e4-4b15-a5fc-65287de4ff31
f(x, y) = fyl([x, y], polytope[y_index]; polytope);

# ╔═╡ 58e2706a-477c-41b1-b52b-11369a5a9ef8
Z = @. f(X', Y);

# ╔═╡ 8163d699-ce03-4e12-ad54-d70f8eeaf283
TwoColumn(set_nb_samples_perturbed, set_epsilon_perturbed)

# ╔═╡ 7ffc8bc9-0e80-4805-8c96-4b672d77a3c3
contour(X, Y, Z; color=:turbo, fill=true, xlabel="θ₁", ylabel="θ₂")

# ╔═╡ 59451577-1200-4fc7-bb53-d7d5bd06bd03
fyl([0.0, 0.0], polytope[y_index]; polytope)

# ╔═╡ 9c67de2a-bd84-44e8-bc50-8760829581c2
minimum(Z)

# ╔═╡ e0363b40-2ba8-4af6-abeb-fdc2f183ded1
question_box(md"4. What happens when $\varepsilon = 0$? What happens when $\varepsilon$ increases?")

# ╔═╡ 3a84fd20-41fa-4156-9be5-a0371754b394
md"""
# 2. Pathfinding on Warcraft maps
"""

# ╔═╡ ee87d357-318f-40f1-a82a-fe680286e6cd
md"""
In this section, we define learning pipelines for the Warcraft shortest path problem. 
We have a sub-dataset of Warcraft terrain images (source: [Vlastelica et al. (2020)](https://openreview.net/forum?id=BkevoJSYPB)), corresponding black-box cost functions, and optionally the label shortest path solutions and cell costs. 
We want to learn the cost of the cells, using a neural network embedding, to predict good shortest paths on new test images.
More precisely, each point in our dataset consists in:
- an image of terrain ``I``.
- a black-box cost function ``c`` to evaluate any given path (optional).
- a label shortest path ``P`` from the top-left to the bottom-right corners (optional). 
- the true cost of each cell of the grid (optional).
We can exploit the images to approximate the true cell costs, so that when considering a new test image of terrain, we predict a good shortest path from its top-left to its bottom-right corners.
The question is: how should we combine these features?
We use `InferOpt` to learn the appropriate costs.

In what follows, we'll build the following pipeline:
"""

# ╔═╡ 5c231f46-02b0-43f9-9101-9eb222cff972
load("./warcraft_pipeline.png")

# ╔═╡ 94192d5b-c4e9-487f-a36d-0261d9e86801
md"""
## I - Dataset and plots
"""

# ╔═╡ 98eb10dd-a4a1-4c91-a0cd-dd1d1e6bc89a
md"""
We first give the path of the dataset folder:
"""

# ╔═╡ 8d2ac4c8-e94f-488e-a1fa-611b7b37fcea
decompressed_path = joinpath(".", "data")

# ╔═╡ 4e2a1703-5953-4901-b598-9c1a98a5fc2b
md"""
### a) Gridgraphs
"""

# ╔═╡ 6d1545af-9fd4-41b2-9a9b-b4472d6c360e
md"""For the purposes of this TP, we consider grid graphs, as implemented in [GridGraphs.jl](https://github.com/gdalle/GridGraphs.jl).
In such graphs, each vertex corresponds to a couple of coordinates ``(i, j)``, where ``1 \leq i \leq h`` and ``1 \leq j \leq w``.
"""

# ╔═╡ e2c4292f-f2e8-4465-b3e3-66be158cacb5
h, w = 12, 12;

# ╔═╡ bd7a9013-199a-4bec-a5a4-4165da61f3cc
g = GridGraph(rand(h, w); directions=QUEEN_DIRECTIONS)

# ╔═╡ 03fd2b72-795f-4cc4-9713-8d2fe2da5429
g.vertex_weights

# ╔═╡ c04157e6-52a9-4d2e-add8-680dc71e5aaa
md"""For convenience, `GridGraphs.jl` also provides custom functions to compute shortest paths efficiently. We use the Dijkstra implementation.
Let us see what those paths look like.
"""

# ╔═╡ 9fae85ed-8bbb-4827-aaee-ec7b11a3bf7b
grid_dijkstra(g, 1, nv(g))

# ╔═╡ 6d78e17c-df70-4b40-96c3-84e0ebcf5063
grid_bellman_ford(g, 1, nv(g))

# ╔═╡ 2cca230e-8008-4924-a9a2-78f35f0d6a42
p = path_to_matrix(g, grid_dijkstra(g, 1, nv(g)))

# ╔═╡ 3044c025-bfb4-4563-8563-42a783e625e2
md"""
### b) Dataset functions
"""

# ╔═╡ 6d21f759-f945-40fc-aaa3-7374470c4ef0
md"""
The first dataset function `read_dataset` is used to read the images, cell costs and shortest path labels stored in files of the dataset folder.
"""

# ╔═╡ 3c141dfd-b888-4cf2-8304-7282aabb5aef
"""
	read_dataset(decompressed_path::String, dtype::String="train")

Read the dataset of type `dtype` at the `decompressed_path` location.
The dataset is made of images of Warcraft terrains, cell cost labels and shortest path labels.
They are returned separately, with proper axis permutation and image scaling to be consistent with 
`Flux` embeddings.
"""
function read_dataset(decompressed_path::String, dtype::String="train")
    # Open files
    data_dir = joinpath(decompressed_path, "warcraft_shortest_path_oneskin", "12x12")
    data_suffix = "maps"
    terrain_images = npzread(joinpath(data_dir, dtype * "_" * data_suffix * ".npy"))
    terrain_weights = npzread(joinpath(data_dir, dtype * "_vertex_weights.npy"))
    terrain_labels = npzread(joinpath(data_dir, dtype * "_shortest_paths.npy"))
    # Reshape for Flux
    terrain_images = permutedims(terrain_images, (2, 3, 4, 1))
    terrain_labels = permutedims(terrain_labels, (2, 3, 1))
    terrain_weights = permutedims(terrain_weights, (2, 3, 1))
    # Normalize images
    terrain_images = Array{Float32}(terrain_images ./ 255)
    println("Train images shape: ", size(terrain_images))
    println("Train labels shape: ", size(terrain_labels))
    println("Weights shape:", size(terrain_weights))
    return terrain_images, terrain_labels, terrain_weights
end

# ╔═╡ c18d4b8f-2ae1-4fde-877b-f53823a42ab1
md"""
Once the files are read, we want to give an adequate format to the dataset, so that we can easily load samples to train and test models. The function `create_dataset` therefore calls the previous `read_dataset` function: 
"""

# ╔═╡ 8c8bb6a1-12cd-4af3-b573-c22383bdcdfb
"""
	create_dataset(decompressed_path::String, nb_samples::Int=10000)

Create the dataset corresponding to the data located at `decompressed_path`, possibly sub-sampling `nb_samples` points.
The dataset is made of images of Warcraft terrains, cell cost labels and shortest path labels.
It is a `Vector` of tuples, each `Tuple` being a dataset point.
"""
function create_dataset(decompressed_path::String, nb_samples::Int=10000)
    terrain_images, terrain_labels, terrain_weights = read_dataset(
        decompressed_path, "train"
    )
    X = [
        reshape(terrain_images[:, :, :, i], (size(terrain_images[:, :, :, i])..., 1)) for
        i in 1:nb_samples
    ]
    Y = [terrain_labels[:, :, i] for i in 1:nb_samples]
    WG = [terrain_weights[:, :, i] for i in 1:nb_samples]
    return collect(zip(X, Y, WG))
end

# ╔═╡ 4a9ed677-e294-4194-bf32-9580d1e47bda
md"""
Last, as usual in machine learning implementations, we split a dataset into train and test sets. The function `train_test_split` does the job:

"""

# ╔═╡ 0514cde6-b425-4fe7-ac1e-2678b64bbee5
"""
	train_test_split(X::AbstractVector, train_percentage::Real=0.5)

Split a dataset contained in `X` into train and test datasets.
The proportion of the initial dataset kept in the train set is `train_percentage`.
"""
function train_test_split(X::AbstractVector, train_percentage::Real=0.5)
    N = length(X)
    N_train = floor(Int, N * train_percentage)
    N_test = N - N_train
    train_ind, test_ind = 1:N_train, (N_train+1):(N_train+N_test)
    X_train, X_test = X[train_ind], X[test_ind]
    return X_train, X_test
end

# ╔═╡ caf02d68-3418-4a6a-ae25-eabbbc7cae3f
md"""
### c) Plot functions
"""

# ╔═╡ 61db4159-84cd-4e3d-bc1e-35b35022b4be
md"""
In the following cell, we define utility plot functions to have a glimpse at images, cell costs and paths. Their implementation is not at the core of this tutorial, they are thus hidden.
"""

# ╔═╡ 08ea0d7e-2ffe-4f2e-bd8c-f15f9af0f35b
begin
    """
        convert_image_for_plot(image::Array{Float32,3})::Array{RGB{N0f8},2}
    Convert `image` to the proper data format to enable plots in Julia.
    """
    function convert_image_for_plot(image::Array{Float32,3})::Array{RGB{N0f8},2}
        new_img = Array{RGB{N0f8},2}(undef, size(image)[1], size(image)[2])
        for i = 1:size(image)[1]
            for j = 1:size(image)[2]
                new_img[i, j] = RGB{N0f8}(image[i, j, 1], image[i, j, 2], image[i, j, 3])
            end
        end
        return new_img
    end

    """
    	plot_image_weights_path(;im, weights, path)
    Plot the image `im`, the weights `weights`, and the path `path` on the same Figure.
    """
    function plot_image_weights_path(x, y, θ; θ_title="Weights", y_title="Path", θ_true=θ)
        im = dropdims(x; dims=4)
        img = convert_image_for_plot(im)
        p1 = Plots.plot(
            img;
            aspect_ratio=:equal,
            framestyle=:none,
            size=(300, 300),
            title="Terrain image"
        )
        p2 = Plots.heatmap(
            θ;
            yflip=true,
            aspect_ratio=:equal,
            framestyle=:none,
            padding=(0., 0.),
            size=(300, 300),
            legend=false,
            title=θ_title,
            clim=(minimum(θ_true), maximum(θ_true))
        )
        p3 = Plots.plot(
            Gray.(y .* 0.7);
            aspect_ratio=:equal,
            framestyle=:none,
            size=(300, 300),
            title=y_title
        )
        plot(p1, p2, p3, layout=(1, 3), size=(900, 300))
    end

    """
        plot_loss_and_gap(losses::Matrix{Float64}, gaps::Matrix{Float64},  options::NamedTuple; filepath=nothing)

    Plot the train and test losses, as well as the train and test gaps computed over epochs.
    """
    function plot_loss_and_gap(losses::Matrix{Float64}, gaps::Matrix{Float64}; filepath=nothing)
        p1 = plot(collect(1:nb_epochs), losses, title="Loss", xlabel="epochs", ylabel="loss", label=["train" "test"])
        p2 = plot(collect(0:nb_epochs), gaps, title="Gap", xlabel="epochs", ylabel="ratio", label=["train" "test"])
        pl = plot(p1, p2, layout=(1, 2))
        isnothing(filepath) || Plots.savefig(pl, filepath)
        return pl
    end
end;

# ╔═╡ d58098e8-bba5-445c-b1c3-bfb597789916
md"""
### d) Import and explore the dataset
"""

# ╔═╡ a0644bb9-bf62-46aa-958e-aeeaaba3482e
md"""
Once we have both defined the functions to read and create a dataset, and to visualize it, we want to have a look at images and paths. Before that, we set the size of the dataset, as well as the train proportion: 
"""

# ╔═╡ eaf0cf1f-a7be-4399-86cc-66c131a57e44
nb_samples, train_prop = 100, 0.8;

# ╔═╡ 2470f5ab-64d6-49d5-9816-0c958714ca73
info(md"We focus only on $nb_samples dataset points, and use a $(trunc(Int, train_prop*100))% / $(trunc(Int, 100 - train_prop*100))% train/test split.")

# ╔═╡ 73bb8b94-a45f-4dbb-a4f6-1f25ad8f194c
begin
    dataset = create_dataset(decompressed_path, nb_samples)
    train_dataset, test_dataset = train_test_split(dataset, train_prop)
end;

# ╔═╡ c9a05a6e-90c3-465d-896c-74bbb429f66a
md"""
We can have a glimpse at the dataset, use the slider to visualize each tuple (image, weights, label path).
"""

# ╔═╡ fd83cbae-638e-49d7-88da-588fe055c963
md"""
``n =`` $(@bind n Slider(1:length(dataset); default=1, show_value=true))
"""

# ╔═╡ 828869da-0a1f-4a26-83ba-78e7a31f5eb9
plot_image_weights_path(dataset[n]...)

# ╔═╡ fa62a7b3-8f17-42a3-8428-b2ac7eae737a
md"""
## II - Combinatorial functions
"""

# ╔═╡ 0f299cf1-f729-4999-af9d-4b39730100d8
md"""
We focus on additional optimization functions to define the combinatorial layer of our pipelines.
"""

# ╔═╡ e59b06d9-bc20-4d70-8940-5f0a53389738
md"""
### a) Recap on the shortest path problem
"""

# ╔═╡ 75fd015c-335a-481c-b2c5-4b33ca1a186a
md"""
Let $D = (V, A)$ be a digraph, $(c_a)_{a \in A}$ the cost associated to the arcs of the digraph, and $(o, d) \in V^2$ the origin and destination nodes. The problem we consider is the following:

**Shortest path problem:** Find an elementary path $P$ from node $o$ to node $d$ in the digraph $D$ with minimum cost $c(P) = \sum_{a \in P} c_a$.
"""

# ╔═╡ dfac541d-a1fe-4822-9bc4-06d1a4f4ec6a
question_box(md"5. When the cost function is non-negative, which algorithm can we use ?")

# ╔═╡ 4050b2c4-628c-4647-baea-c50236558712
question_box(md"6. In the case the graph contains no absorbing cycle, which algorithm can we use ? 	On which principle is it based ?")

# ╔═╡ 654066dc-98fe-4c3b-92a9-d09efdfc8080
md"""
In the following, we will perturb or regularize the output of a neural network to define the candidate cell costs to predict shortest paths. We therefore need to deal with possibly negative costs.
"""

# ╔═╡ 9f902433-9a21-4b2d-b5d7-b18a04bf6022
question_box(md"7. In the general case, can we fix the maximum length of a feasible solution of the shortest path problem ? How ? Can we derive an dynamic programming algorithm based on this ?")

# ╔═╡ dc359052-19d9-4f29-903c-7eb9b210cbcd
md"""
###  b) From shortest path to generic maximizer
"""

# ╔═╡ b93009a7-533f-4c5a-a4f5-4c1d88cc1be4
md"""
Now that we have defined and implemented an algorithm to deal with the shortest path problem, we wrap it in a maximizer function to match the generic framework of structured prediction.

The maximizer needs to take predicted weights `θ` as their only input, and can take some keyword arguments if needed (some instance information for example).
"""

# ╔═╡ 20999544-cefd-4d00-a68c-cb6cfea36b1a
function dijkstra_maximizer(θ::AbstractMatrix; kwargs...)
    g = GridGraph(-θ; directions=QUEEN_DIRECTIONS)
    path = grid_dijkstra(g, 1, nv(g))
    y = path_to_matrix(g, path)
    return y
end

# ╔═╡ 2c78fd8f-2a34-4307-8762-b6d636fa26f0
"""
    grid_bellman_ford_warcraft(g, s, d, length_max)

Apply the Bellman-Ford algorithm on an `GridGraph` `g`, and return a `ShortestPathTree` with source `s` and destination `d`,
among the paths having length smaller than `length_max`.
"""
function grid_bellman_ford_warcraft(g::GridGraph{T,R,W,A}, s::Integer, d::Integer, length_max::Int=nv(g)) where {T,R,W,A}
    # Init storage
    parents = zeros(Int, nv(g), length_max + 1)
    dists = fill(Inf, nv(g), length_max + 1)
    # Add source
    dists[s, 1] = zero(T)
    # Main loop
    for k in 1:length_max
        for v in vertices(g)
            for u in inneighbors(g, v)
                d_u = dists[u, k]
                if !isinf(d_u)
                    d_v = dists[v, k+1]
                    d_v_through_u = d_u + GridGraphs.vertex_weight(g, v)
                    if isinf(d_v) || (d_v_through_u < d_v)
                        dists[v, k+1] = d_v_through_u
                        parents[v, k+1] = u
                    end
                end
            end
        end
    end
    # Get length of the shortest path
    k_short = argmin(dists[d, :])
    if isinf(dists[d, k_short])
        println("No shortest path with less than $length_max arcs")
        return Int[]
    end
    # Deduce the path
    v = d
    path = [v]
    k = k_short
    while v != s
        v = parents[v, k]
        if v == 0
            return Int[]
        else
            pushfirst!(path, v)
            k = k - 1
        end
    end
    return path
end

# ╔═╡ b2ea7e31-82c6-4b01-a8c6-26c3d7a2d562
function bellman_maximizer(θ::AbstractMatrix; kwargs...)
    g = GridGraph(-θ; directions=QUEEN_DIRECTIONS)
    path = grid_bellman_ford_warcraft(g, 1, nv(g))
    y = path_to_matrix(g, path)
    return y
end

# ╔═╡ 927147a9-6308-4b84-9688-ddcdf09c83d0
danger(md"`InferOpt.jl` wrappers only take maximization algorithms as input. Don't forget to change some signs if your solving a minimization problem instead.")

# ╔═╡ 76d4caa4-a10c-4247-a624-b6bfa5a743bc
md"""
!!! info "The maximizer function will depend on the pipeline"
	Note that we use the function `grid_dijkstra` already implemented in the `GridGraphs.jl` package when we deal with non-negative cell costs. In the following, we will use either Dijkstra or Ford-Bellman algorithm depending on the learning pipeline. You will have to modify the maximizer function to use depending on the experience you do.
"""

# ╔═╡ 91ec470d-f2b5-41c1-a50f-fc337995c73f
md"""
## III - Learning functions
"""

# ╔═╡ f899c053-335f-46e9-bfde-536f642700a1
md"""
### a) Convolutional neural network: predictor for the cost vector
"""

# ╔═╡ 6466157f-3956-45b9-981f-77592116170d
md"""
We implement several elementary functions to define our machine learning predictor for the cell costs.
"""

# ╔═╡ 211fc3c5-a48a-41e8-a506-990a229026fc
"""
    average_tensor(x)

Average the tensor `x` along its third axis.
"""
function average_tensor(x)
    return sum(x, dims=[3]) / size(x)[3]
end

# ╔═╡ 7b8b659c-9c7f-402d-aa7b-63c17179560e
"""
    neg_tensor(x)

Compute minus softplus element-wise on tensor `x`.
"""
function neg_tensor(x)
    return -softplus.(x)
end

# ╔═╡ e392008f-1a92-4937-8d8e-820211e44422
"""
    squeeze_last_dims(x)

Squeeze two last dimensions on tensor `x`.
"""
function squeeze_last_dims(x)
    return reshape(x, size(x, 1), size(x, 2))
end

# ╔═╡ 8f23f8cc-6393-4b11-9966-6af67c6ecd40
md"""
!!! info "CNN as predictor"
	The following function defines the convolutional neural network we will use as cell costs predictor.
"""

# ╔═╡ 51a44f11-646c-4f1a-916e-6c83750f8f20
"""
    create_warcraft_embedding()

Create and return a `Flux.Chain` embedding for the Warcraft terrains, inspired by [differentiation of blackbox combinatorial solvers](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers/blob/master/models.py).

The embedding is made as follows:
1) The first 5 layers of ResNet18 (convolution, batch normalization, relu, maxpooling and first resnet block).
2) An adaptive maxpooling layer to get a (12x12x64) tensor per input image.
3) An average over the third axis (of size 64) to get a (12x12x1) tensor per input image.
4) The element-wize `neg_tensor` function to get cell weights of proper sign to apply shortest path algorithms.
5) A squeeze function to forget the two last dimensions. 
"""
function create_warcraft_embedding()
    resnet18 = ResNet(18; pretrain=false, nclasses=1)
    model_embedding = Chain(
        resnet18.layers[1][1][1],
        resnet18.layers[1][1][2],
        resnet18.layers[1][1][3],
        resnet18.layers[1][2][1],
        AdaptiveMaxPool((12, 12)),
        average_tensor,
        neg_tensor,
        squeeze_last_dims,
    )
    return model_embedding
end

# ╔═╡ d793acb0-fd30-48ba-8300-dff9caac536a
md"""
We can build the encoder this way:
"""

# ╔═╡ d9f5281b-f34b-485c-a781-804b8472e38c
initial_encoder = create_warcraft_embedding()

# ╔═╡ 9782f5fb-7e4b-4d8a-a77a-e4f5b9a71ab5
md"""
### b) Loss and gap utility functions
"""

# ╔═╡ 596734af-cf81-43c9-a525-7ea88a209a53
md"""
In the cell below, we define the `cost` function seen as black-box to evaluate the cost of a given path on the grid, given the true costs `c_true`.
"""

# ╔═╡ 0ae90d3d-c718-44b2-81b5-25ce43f42988
cost(y; c_true) = dot(y, c_true)

# ╔═╡ 6a482757-8a04-4724-a3d2-33577748bd4e
md"""
During training, we want to evaluate the quality of the predicted paths, both on the train and test datasets. We define the shortest path cost ratio between a candidate shortest path $\hat{y}$ and the label shortest path $y$ as: $r(\hat{y},y) = c(\hat{y}) / c(y)$.
"""

# ╔═╡ a47a12b4-976e-4250-9e19-a99f915556af
question_box(md"8. What is the link in our problem between the shortest path cost ratio and the gap of a given solution with respect to the optimal solution ?")

# ╔═╡ 9eb0ca01-bd65-48df-ab32-beaca2e38482
md"""
!!! info
	The following code defines the `shortest_path_cost_ratio` function. The candidate path $\hat{y}$ is given by the output of `model` applied on image `x`, and `y` is the target shortest path.
"""

# ╔═╡ 26c71a94-5b30-424f-8242-c6510d41bb52
"""
	shortest_path_cost_ratio(model, x, y, kwargs)
Compute the ratio between the cost of the solution given by the `model` cell costs and the cost of the true solution.
We evaluate both the shortest path with respect to the weights given by `model(x)` and the labelled shortest path `y`
using the true cell costs stored in `kwargs.wg.weights`. 
This ratio is by definition greater than one. The closer it is to one, the better is the solution given by the current 
weights of `model`. We thus track this metric during training.
"""
function shortest_path_cost_ratio(model, x, y_true, θ_true; maximizer)
    θ = model(x)
    y = maximizer(θ)
    return dot(θ_true, y) / dot(θ_true, y_true)
end

# ╔═╡ dd1791a8-fa59-4a36-8794-fccdcd7c912a
"""
	shortest_path_cost_ratio(model, batch)
Compute the average cost ratio between computed and true shorest paths over `batch`. 
"""
function shortest_path_cost_ratio(model, batch; maximizer)
    return sum(shortest_path_cost_ratio(model, item[1], item[2], item[3]; maximizer) for item in batch) / length(batch)
end

# ╔═╡ 633e9fea-fba3-4fe6-bd45-d19f89cb1808
"""
	shortest_path_cost_gap(; model, dataset)
Compute the average cost ratio between computed and true shorest paths over `dataset`. 
"""
function shortest_path_cost_gap(; model, dataset, maximizer)
    return (sum(shortest_path_cost_ratio(model, batch; maximizer) for batch in dataset) / length(dataset) - 1) * 100
end

# ╔═╡ 8c8b514e-8478-4b2b-b062-56832115c670
md"""
### c) Main training function
"""

# ╔═╡ 93dd97e6-0d37-4d94-a3f6-c63dc856fa66
md"""
We now consider the generic learning function. We want to minimize a given `flux_loss` over the `train_dataset`, by updating the parameters of `encoder`. We do so using `Flux.jl` package which contains utility functions to backpropagate in a stochastic gradient descent framework. We also track the loss and cost ratio metrics both on the train and test sets. The hyper-parameters are stored in the `options` tuple. 
"""

# ╔═╡ d35f0e8b-6634-412c-b5f3-ffd11246276c
md"""
The following block defines the generic learning function.
"""

# ╔═╡ 920d94cd-bfb5-4c02-baa3-f346d5c95e2e
md"""
## IV - Pipelines
"""

# ╔═╡ 658bd4b9-ee97-4b81-9337-ee6d1ccdf7bb
md"""
!!! info "Preliminary remark"
	Here come the specific learning experiments. The following code cells will have to be modified to deal with different settings.
"""

# ╔═╡ f1b50452-4e8c-4393-b112-7a4cfb3b7fb4
md"""
As you know, the solution of a linear program is not differentiable with respect to its cost vector. Therefore, we need additional tricks to be able to update the parameters of the CNN defined by `create_warcraft_embedding`. Two points of view can be adopted: perturb or regularize the maximization problem. They can be unified when introducing probabilistic combinatorial layers, detailed in this [paper](https://arxiv.org/pdf/2207.13513.pdf). They are used in two different frameworks:

- Learning by imitation when we have target shortest path examples in the dataset.
- Learning by experience when we only have access to the images and to a black-box cost function to evaluate any candidate path.

In this section, we explore different combinatorial layers, as well as the learning by imitation and learning by experience settings.
"""

# ╔═╡ 9a670af7-cc20-446d-bf22-4e833cc9d854
md"""
### a) Learning by imitation with additive perturbation
"""

# ╔═╡ b389a6a0-dc8e-4c6f-8a82-4f8878ffe879
md"""
#### 1) Hyperparameters
"""

# ╔═╡ e0e97839-884a-49ed-bee4-f1f2ace5f5e0
md"""
We first define the hyper-parameters for the learning process. They include:
- The regularization size $\varepsilon$.
- The number of samples drawn for the approximation of the expectation $M$.
- The number of learning epochs `nb_epochs`.
- The batch size for the stochastic gradient descent `batch_size`.
- The starting learning rate for ADAM optimizer `lr_start`.
"""

# ╔═╡ bcdd60b8-e0d8-4a70-88d6-725269447c9b
begin
    ε = 0.1
    M = 10
    nb_epochs = 20
    batch_size = 80
    lr_start = 1e-3
end;

# ╔═╡ a6a56523-90c9-40d2-9b68-26e20c1a5527
function train_function!(;
    encoder, loss, train_data, test_data, lr_start, nb_epoch, batch_size, maximizer
)
    # batch stuff
    batch_loss(batch) = sum(loss(item...) for item in batch)
    train_dataset = Flux.DataLoader(train_data; batchsize=batch_size)
    test_dataset = Flux.DataLoader(test_data; batchsize=length(test_data))

    # Store the train loss and gap metric
    losses = Matrix{Float64}(undef, nb_epochs, 2)
    cost_gaps = Matrix{Float64}(undef, nb_epochs + 1, 2)

    # Optimizer
    opt = Adam(lr_start)

    # model parameters
    par = Flux.params(encoder)

    cost_gaps[1, 1] = shortest_path_cost_gap(; model=encoder, dataset=train_dataset, maximizer)
    cost_gaps[1, 2] = shortest_path_cost_gap(; model=encoder, dataset=test_dataset, maximizer)

    # Train loop
    @progress "Training epoch: " for epoch in 1:nb_epochs
        train_loss = 0.0
        for batch in train_dataset
            loss_value = 0
            gs = gradient(par) do
                loss_value = batch_loss(batch)
            end
            train_loss += loss_value
            Flux.update!(opt, par, gs)
        end

        # compute and store epoch metrics
        losses[epoch, 1] = train_loss / (nb_samples * train_prop)
        losses[epoch, 2] = sum([batch_loss(batch) for batch in test_dataset]) / (nb_samples * (1 - train_prop))
        cost_gaps[epoch+1, 1] = shortest_path_cost_gap(; model=encoder, dataset=train_dataset, maximizer)
        cost_gaps[epoch+1, 2] = shortest_path_cost_gap(; model=encoder, dataset=test_dataset, maximizer)
    end
    return losses, cost_gaps, deepcopy(encoder)
end

# ╔═╡ 677f20f9-61ed-46ed-af65-73af73f7af7d
tip(md"Feel free to play around with hyperparameters, observe and report their impact on the training performances.")

# ╔═╡ 9de99f4a-9970-4be1-9e16-e64ed4e10277
md"""
#### 2) Specific pipeline
"""

# ╔═╡ 518e7077-d61b-4f60-987f-d556e3eb1d0d
md"""
!!! info "What is a pipeline ?"
	This portion of code is the crucial part to define the learning pipeline. It contains: 
	- an encoder, the machine learning predictor, in our case a CNN.
	- a maximizer possibly applied to the output of the encoder before computing the loss.
	- a differentiable loss to evaluate the quality of the output of the pipeline.
	
	Its definition depends on the learning setting we consider.
"""

# ╔═╡ 6ed223f4-de31-43f7-a16f-16523c1d61ea
md"As already seen in the previous sections, we wrap our shortest path algorithm in a `PerturbedAdditive`"

# ╔═╡ 8b2bd08c-866a-4c6e-a2fc-261dc8c05f2a
chosen_maximizer = bellman_maximizer

# ╔═╡ 73607123-a784-483e-9241-772e5937d59d
perturbed_maximizer = PerturbedAdditive(chosen_maximizer; ε=ε, nb_samples=M)

# ╔═╡ 13945989-fb32-4027-a67b-e2a9a9254446
md"And define the associated Fenchel Young loss:"

# ╔═╡ 2e926dc4-0f12-411e-85e2-5dcffdcc1266
loss = FenchelYoungLoss(perturbed_maximizer)

# ╔═╡ b7674b98-526c-40ee-bf83-a9d6e7be6e4f
encoder = deepcopy(initial_encoder)

# ╔═╡ f5e789b2-a62e-4818-90c3-76f39ea11aaa
md"""
#### 3) Flux loss definition
"""

# ╔═╡ efa7736c-22c0-410e-94da-1df315f22bbf
md"""
From the generic definition of the pipeline we define a loss function compatible with `Flux.jl` package. Its definition depends on the learning setting we consider.
In this subsection, we are in a learning by imitation setting
"""

# ╔═╡ 9b351fbd-2820-4353-b3fa-ec0e6d07d861
imitation_flux_loss(x, y, θ) = loss(encoder(x), y)

# ╔═╡ b4451d05-1ac5-4962-88d8-e59d9ca225ea
warning_box(md"If you want to use the `train_function!` generic function defined above, the loss needs to take as argument x, y and θ in this order, even if it does not use all of them.")

# ╔═╡ 58b7267d-491d-40f0-b4ba-27ed0c9cc855
md"""
#### 4) Apply the learning function
"""

# ╔═╡ ac76b646-7c28-4384-9f04-5e4de5df154f
md"""
Given the specific pipeline and loss, we can apply our generic train function to update the weights of the CNN predictor.
"""

# ╔═╡ effbea0f-e9af-469c-a792-10078da46b39
danger(md"Click the checkbox to activate the training cell $(@bind train CheckBox()) 

It may take some time to run and affect the reactivity of the notebook. Then you can read what follows. For the numerical experiments, you transfer the code into a regular julia script if you prefer.")

# ╔═╡ 83a14158-33d1-4f16-85e1-2726c8fbbdfc
loss_history, gap_history, final_encoder = train ? train_function!(;
    encoder=encoder,
    maximizer=chosen_maximizer,
    loss=imitation_flux_loss,
    train_data=train_dataset,
    test_data=test_dataset,
    lr_start=lr_start,
    batch_size=batch_size,
    nb_epoch=nb_epochs
) : (zeros(nb_epochs, 2), zeros(nb_epochs + 1, 2), encoder);

# ╔═╡ 4b31dca2-0195-4899-8a3a-e9772fabf495
md"""
#### 5) Plot results
"""

# ╔═╡ 79e0deab-1e36-4863-ad10-187ed8555c72
md"""
Loss and gap over epochs, train and test datasets.
"""

# ╔═╡ 66d385ba-9c6e-4378-b4e0-e54a4df346a5
plot_loss_and_gap(loss_history, gap_history)

# ╔═╡ 76b7002a-5df0-4d77-b03f-6b677cc50de4
loss_history

# ╔═╡ db799fa2-0e48-43ee-9ee1-80ff8d2e5de7
md"""
To assess performance, we can compare the true and predicted paths.
"""

# ╔═╡ eb3a6009-e181-443c-bb77-021e867030e4
md"""
!!! info "Visualize the model performance"
	We now want to see the effect of the learning process on the predicted costs and shortest paths. Use the slider to swipe through datasets.
"""

# ╔═╡ 01a1fd52-ff6c-44c6-ab1a-d1c141a4d54e
TwoColumn(md"Choose dataset you want to evaluate on:", md"""data = $(@bind data Select([train_dataset => "train", test_dataset => "test"]))""")

# ╔═╡ 521f5ffa-2c22-44c5-8bdb-67410431ca2e
begin
    test_predictions = []
    dataset_to_test = data
    for (x, y_true, θ_true) in dataset_to_test
        θ₀ = initial_encoder(x)
        y₀ = UInt8.(chosen_maximizer(θ₀))
        θ = final_encoder(x)
        y = UInt8.(chosen_maximizer(θ))
        push!(test_predictions, (; x, y_true, θ_true, θ₀, y₀, θ, y))
    end
end

# ╔═╡ f9b35e98-347f-4ebd-a690-790c7b0e03d8
md"""
``j =`` $(@bind j Slider(1:length(dataset_to_test); default=1, show_value=true))
"""

# ╔═╡ a828548f-175b-4cf0-b0aa-d9eef0477f4d
(; x, y_true, θ_true, θ₀, y₀, θ, y) = test_predictions[j]

# ╔═╡ a1043a1c-5840-4175-aa4a-ef432c353073
plot_image_weights_path(x, y_true, θ_true)

# ╔═╡ b1a44835-58d7-462f-a0d4-85bc02d3fdc6
md"Predictions of the trained neural network:"

# ╔═╡ 91e520aa-97a1-40b2-8936-93c93a63011c
plot_image_weights_path(
    x, y, -θ; θ_title="Predicted weights", y_title="Predicted path", θ_true=θ_true
)

# ╔═╡ 39daeb26-66d6-4a05-979f-76666444c73b
md"Predictions of the initial untrained neural network:"

# ╔═╡ 0e8ea002-6bc8-4684-a72a-f7d7062eecc0
plot_image_weights_path(
    x, y₀, -θ₀; θ_title="Initial predicted weights", y_title="Initial predicted path", θ_true=θ_true
)

# ╔═╡ 9a9b3942-72f2-4c9e-88a5-af927634468c
md"""
### b) Learning by imitation with multiplicative perturbation
"""

# ╔═╡ 1ff198ea-afd5-4acc-bb67-019051ff149b
md"""
We introduce a variant of the additive pertubation defined above, which is simply based on an element-wise product $\odot$:
"""

# ╔═╡ 44ece9ce-f9f1-46f3-90c6-cb0502c92c67
md"""
${y}_\varepsilon^\odot (\theta) := \mathbb{E}_Z \bigg[\operatorname{argmax}_{y \in \mathcal{C}} (\theta \odot e^{\epsilon Z - \varepsilon^2 \mathbf{1} / 2})^\top y \bigg]$
"""

# ╔═╡ 5fe95aa5-f670-4329-a933-240a8c074dea
question_box(md"9. What is the advantage of this perturbation compared with the additive one in terms of combinatorial problem ? Which algorithm can we use to compute shortest paths ?")

# ╔═╡ 43d68541-84a5-4a63-9d8f-43783cc27ccc
md"We omit the details of the loss derivations and concentrate on implementation."

# ╔═╡ 5c6d39b0-9942-4173-9455-39cb3c174873
TODO("Implement the training similarly to previous subsection, by using a multiplicative perturbation instead of the additive one.")

# ╔═╡ 99468dd9-4b97-48e6-803b-489dc1cefdf8
hint(md"You can modify the previous additive implementation below, by replacing the `PerturbedAdditive` regularization with a `PerturbedMultiplicative` one. You can also modify use `dijkstra_maximizer` instead of `belmann_maximizer` as the CO algorithm, which runs faster.")

# ╔═╡ 0a0e7b32-e1f4-4d5c-8ebc-b5d06b61e6df


# ╔═╡ f6d87e32-419a-48be-8054-f54fb6e4cef3
question_box(md"10. Comment your experiments and results (some figures/tables may be useful)")

# ╔═╡ 0fd29811-9e17-4c97-b9b7-ec9cc51b435f
md"""
### c) Smart Predict then optimize
"""

# ╔═╡ 71726572-0341-4344-9a3f-410d3bbc430a
TODO(md"Replace the `FenchelYoungLoss` by a `SPOPlusLoss` in order to leverage the knowledge about the true costs in the train dataset.")

# ╔═╡ 7ccf487c-49a1-49e9-bc19-e2f4e8a7d331
hint(md"You can replace the `FenchelYoungLoss` by `SPOPlusLoss(true_maximizer)`, we do not need to use the `Perturbed` here.")

# ╔═╡ e6f45063-e553-42ab-8344-69ff78ab520f


# ╔═╡ 8bb55d7e-1817-4b81-8de6-ad31191d08e8
question_box(md"11. Comment your experiments and results (some figures/tables may be useful).")

# ╔═╡ 90a47e0b-b911-4728-80b5-6ed74607833d
md"""
### d) Learning by experience with multiplicative perturbation
"""

# ╔═╡ 5d79b8c1-beea-4ff9-9830-0f5e1c4ef29f
md"""
When we restrict the train dataset to images $I$ and black-box cost functions $c$, we can not learn by imitation. We can instead derive a surrogate version of the regret that is differentiable (see Section 4.1 of this [paper](https://arxiv.org/pdf/2207.13513.pdf)).
"""

# ╔═╡ 418755cb-765f-4a8c-805d-ceac36c7706c
TODO(md"Modify the code above to learn by experience using a multiplicative perturbation and the black-box cost function.")

# ╔═╡ 4625c35f-cd2d-4883-838a-57276e83d241
hint(md"Use the `PushForward` wrapper to define a learn by experience loss.")

# ╔═╡ 0ddf43c5-6ba0-4d22-80ad-6ca8cf92f69a


# ╔═╡ 00cb431b-dced-45fd-a191-12bd59a096f5
question_box(md"12. Comment your experiments and results.")

# ╔═╡ a5bfd185-aa77-4e46-a6b6-d43c4785a7fa
md"""
### e) Learning by experience with half square norm regularization (bonus). 
"""

# ╔═╡ a7b6ecbd-1407-44dd-809e-33311970af12
md"""
For the moment, we have only considered perturbations to derive meaningful gradient information. We now focus on a half square norm regularization.
"""

# ╔═╡ a96e6942-06ab-42d3-a7e5-9c431a676d15
TODO(md"Based on the functions `scaled_half_square_norm` and `grad_scaled_half_square_norm`, use the `RegularizedFrankWolfe` implementation of [`InferOpt.jl`](https://axelparmentier.github.io/InferOpt.jl/dev/algorithms/) to learn by experience. Modify the cells below to do so.", heading="TODO (bonus)")

# ╔═╡ 201ec4fd-01b1-49c4-a104-3d619ffb447b
md"""
The following cell defines the scaled half square norm function and its gradient.
"""

# ╔═╡ 8b544491-b892-499f-8146-e7d1f02aaac1
begin
    scaled_half_square_norm(x, ε=25.) = ε * sum(abs2, x) / 2
    grad_scaled_half_square_norm(x, ε=25.) = ε * identity(x)
    frank_wolfe_kwargs = (;
        line_search=FrankWolfe.Agnostic(),
        max_iteration=20
    )
end;

# ╔═╡ 18fe69ea-2f4b-41d2-b44e-f395af273891


# ╔═╡ b0769c65-9b86-496e-85fc-a8dc43c55576
question_box(md"13. Comment your experiments and results.")

# ╔═╡ Cell order:
# ╟─8b7876e4-2f28-42f8-87a1-459b665cff30
# ╠═e36da6f2-b236-4b67-983c-152a7ff54e05
# ╟─e279878d-9c8d-47c8-9453-3aee1118818b
# ╟─b5b0bb58-9e02-4551-a9ba-0ba0ffceb350
# ╟─b0616d13-41fa-4a89-adb3-bf8b27b13657
# ╟─9adcecda-eaeb-4432-8634-a1ce868c50f5
# ╟─21bee304-8aab-4c57-b3ab-ceec6a608320
# ╟─2067c125-f473-4cc2-a548-87b1b0ad9011
# ╟─d85d1e30-92f4-4bc7-8da9-3b417f51530b
# ╟─7e46ec11-b0ff-4dc7-9939-32ad154aeb96
# ╟─95a43871-924b-4ff1-87ac-76c33d22c9ad
# ╟─269547da-f4ec-4746-9453-5cb8d7703da8
# ╟─68c6b115-5873-4678-9f3a-54b72554e8d3
# ╟─d9dbc402-383a-4aad-9f44-08f06b41ab0d
# ╟─78312b73-42bd-42d3-b31d-83222fd8fbaa
# ╟─4678209f-9bb9-4d3b-b031-575f2fba4916
# ╟─3bb99c85-35de-487d-a5e7-1cd1313fd6ea
# ╟─d447f8af-78de-4306-ba24-22851c366690
# ╟─f5afa452-9485-4dba-93fe-277d87ad0344
# ╟─14af0338-554a-4f71-a290-3b4f16cc6af5
# ╟─2901d761-405a-4800-b1a7-d2a80cf8aea5
# ╟─23f7f158-9a74-4f6b-9718-5609f458b101
# ╟─978b5cff-bd07-48a1-8248-366798bf5d35
# ╟─f13bf21c-33db-4620-add8-bfd82f493d7c
# ╟─c8bf8d9a-783c-41c6-bf33-dc423e249d0b
# ╟─f99d6992-dc3e-41d1-8922-4958886dade2
# ╟─0d20da65-1e53-4b6e-b302-28243c94fb4c
# ╟─87040fd6-bd1a-47dd-875c-2caf5b50d2ce
# ╟─53f7468d-0015-4339-8e27-48812f541329
# ╟─81cd64c4-f317-4555-ab7a-9a5b2b837b91
# ╠═95013865-885c-4e0d-a76b-c2452b10bdad
# ╠═afdd0ea0-054a-4b8e-a7ea-8a21d1e021ff
# ╟─61c624df-384a-4f01-a1c2-20d09d43aa74
# ╠═4b59b997-dc7d-49a9-8557-87d908673c22
# ╟─0fe3676f-70a4-4730-9ce9-ac5bc4204284
# ╟─446cb749-c1ec-46a1-8cff-74a99d0cc2d9
# ╟─7db83f4b-0f9c-4d27-9a5e-bc6aacdae186
# ╟─1ca1d8a3-d7bc-4386-8142-29c5cf2a87a0
# ╠═f370c7c5-0f39-4efa-a298-d913a591412d
# ╟─2773083a-0f6a-4c28-8da3-2a4ee4efdb6f
# ╟─e6efe06c-8833-4a6b-8086-b7ebe91ee703
# ╟─381c10e9-25c2-4ec0-8c37-48f7099abd03
# ╠═0a205ed0-e52d-4017-b78f-23c7447063f3
# ╟─ded783ba-7b43-4351-8951-a452e1e26e3c
# ╟─98c6fffd-26d2-4d94-ba99-1f3a59197079
# ╟─b3fd69fb-a1dd-4102-9ced-eb0566821a57
# ╟─cc978152-4e1a-4aa1-9dc2-7e49f03ead76
# ╟─f2926b6a-1ff0-4157-a6b8-a56f658f4d49
# ╠═712c87ea-91e0-4eaa-807c-6876ee5b311f
# ╟─5dd28e66-afd8-4c9d-bc88-b87e5e13f390
# ╟─6801811b-f68a-43b4-8b78-2f27c0dc6331
# ╟─b748c794-b9b6-4e96-8f65-f34abd6b127e
# ╟─701f4d68-0424-4f9f-b904-84b52f6a4745
# ╠═d64790a7-6a02-44ca-a44f-268fea657690
# ╟─87c4d949-c7d9-4f70-8fe0-f273ad655635
# ╠═e4bf4523-94c2-457f-9dd3-74f100d2dc17
# ╟─e8ce60c1-4981-464a-9a9b-8ac5734a5bb4
# ╠═6b212032-22e4-4b15-a5fc-65287de4ff31
# ╠═58e2706a-477c-41b1-b52b-11369a5a9ef8
# ╟─8163d699-ce03-4e12-ad54-d70f8eeaf283
# ╠═7ffc8bc9-0e80-4805-8c96-4b672d77a3c3
# ╠═59451577-1200-4fc7-bb53-d7d5bd06bd03
# ╠═9c67de2a-bd84-44e8-bc50-8760829581c2
# ╟─e0363b40-2ba8-4af6-abeb-fdc2f183ded1
# ╟─3a84fd20-41fa-4156-9be5-a0371754b394
# ╟─ee87d357-318f-40f1-a82a-fe680286e6cd
# ╟─5c231f46-02b0-43f9-9101-9eb222cff972
# ╟─94192d5b-c4e9-487f-a36d-0261d9e86801
# ╟─98eb10dd-a4a1-4c91-a0cd-dd1d1e6bc89a
# ╠═8d2ac4c8-e94f-488e-a1fa-611b7b37fcea
# ╟─4e2a1703-5953-4901-b598-9c1a98a5fc2b
# ╟─6d1545af-9fd4-41b2-9a9b-b4472d6c360e
# ╠═e2c4292f-f2e8-4465-b3e3-66be158cacb5
# ╠═bd7a9013-199a-4bec-a5a4-4165da61f3cc
# ╠═03fd2b72-795f-4cc4-9713-8d2fe2da5429
# ╟─c04157e6-52a9-4d2e-add8-680dc71e5aaa
# ╠═9fae85ed-8bbb-4827-aaee-ec7b11a3bf7b
# ╠═6d78e17c-df70-4b40-96c3-84e0ebcf5063
# ╠═2cca230e-8008-4924-a9a2-78f35f0d6a42
# ╟─3044c025-bfb4-4563-8563-42a783e625e2
# ╟─6d21f759-f945-40fc-aaa3-7374470c4ef0
# ╟─3c141dfd-b888-4cf2-8304-7282aabb5aef
# ╟─c18d4b8f-2ae1-4fde-877b-f53823a42ab1
# ╟─8c8bb6a1-12cd-4af3-b573-c22383bdcdfb
# ╟─4a9ed677-e294-4194-bf32-9580d1e47bda
# ╟─0514cde6-b425-4fe7-ac1e-2678b64bbee5
# ╟─caf02d68-3418-4a6a-ae25-eabbbc7cae3f
# ╟─61db4159-84cd-4e3d-bc1e-35b35022b4be
# ╟─08ea0d7e-2ffe-4f2e-bd8c-f15f9af0f35b
# ╟─d58098e8-bba5-445c-b1c3-bfb597789916
# ╟─a0644bb9-bf62-46aa-958e-aeeaaba3482e
# ╠═eaf0cf1f-a7be-4399-86cc-66c131a57e44
# ╟─2470f5ab-64d6-49d5-9816-0c958714ca73
# ╠═73bb8b94-a45f-4dbb-a4f6-1f25ad8f194c
# ╟─c9a05a6e-90c3-465d-896c-74bbb429f66a
# ╟─fd83cbae-638e-49d7-88da-588fe055c963
# ╠═828869da-0a1f-4a26-83ba-78e7a31f5eb9
# ╟─fa62a7b3-8f17-42a3-8428-b2ac7eae737a
# ╟─0f299cf1-f729-4999-af9d-4b39730100d8
# ╟─e59b06d9-bc20-4d70-8940-5f0a53389738
# ╟─75fd015c-335a-481c-b2c5-4b33ca1a186a
# ╟─dfac541d-a1fe-4822-9bc4-06d1a4f4ec6a
# ╟─4050b2c4-628c-4647-baea-c50236558712
# ╟─654066dc-98fe-4c3b-92a9-d09efdfc8080
# ╟─9f902433-9a21-4b2d-b5d7-b18a04bf6022
# ╟─dc359052-19d9-4f29-903c-7eb9b210cbcd
# ╟─b93009a7-533f-4c5a-a4f5-4c1d88cc1be4
# ╠═20999544-cefd-4d00-a68c-cb6cfea36b1a
# ╟─2c78fd8f-2a34-4307-8762-b6d636fa26f0
# ╠═b2ea7e31-82c6-4b01-a8c6-26c3d7a2d562
# ╟─927147a9-6308-4b84-9688-ddcdf09c83d0
# ╟─76d4caa4-a10c-4247-a624-b6bfa5a743bc
# ╟─91ec470d-f2b5-41c1-a50f-fc337995c73f
# ╟─f899c053-335f-46e9-bfde-536f642700a1
# ╟─6466157f-3956-45b9-981f-77592116170d
# ╟─211fc3c5-a48a-41e8-a506-990a229026fc
# ╟─7b8b659c-9c7f-402d-aa7b-63c17179560e
# ╟─e392008f-1a92-4937-8d8e-820211e44422
# ╟─8f23f8cc-6393-4b11-9966-6af67c6ecd40
# ╟─51a44f11-646c-4f1a-916e-6c83750f8f20
# ╟─d793acb0-fd30-48ba-8300-dff9caac536a
# ╠═d9f5281b-f34b-485c-a781-804b8472e38c
# ╟─9782f5fb-7e4b-4d8a-a77a-e4f5b9a71ab5
# ╟─596734af-cf81-43c9-a525-7ea88a209a53
# ╠═0ae90d3d-c718-44b2-81b5-25ce43f42988
# ╟─6a482757-8a04-4724-a3d2-33577748bd4e
# ╟─a47a12b4-976e-4250-9e19-a99f915556af
# ╟─9eb0ca01-bd65-48df-ab32-beaca2e38482
# ╟─26c71a94-5b30-424f-8242-c6510d41bb52
# ╟─dd1791a8-fa59-4a36-8794-fccdcd7c912a
# ╟─633e9fea-fba3-4fe6-bd45-d19f89cb1808
# ╟─8c8b514e-8478-4b2b-b062-56832115c670
# ╟─93dd97e6-0d37-4d94-a3f6-c63dc856fa66
# ╟─d35f0e8b-6634-412c-b5f3-ffd11246276c
# ╠═a6a56523-90c9-40d2-9b68-26e20c1a5527
# ╟─920d94cd-bfb5-4c02-baa3-f346d5c95e2e
# ╟─658bd4b9-ee97-4b81-9337-ee6d1ccdf7bb
# ╟─f1b50452-4e8c-4393-b112-7a4cfb3b7fb4
# ╟─9a670af7-cc20-446d-bf22-4e833cc9d854
# ╟─b389a6a0-dc8e-4c6f-8a82-4f8878ffe879
# ╟─e0e97839-884a-49ed-bee4-f1f2ace5f5e0
# ╠═bcdd60b8-e0d8-4a70-88d6-725269447c9b
# ╟─677f20f9-61ed-46ed-af65-73af73f7af7d
# ╟─9de99f4a-9970-4be1-9e16-e64ed4e10277
# ╟─518e7077-d61b-4f60-987f-d556e3eb1d0d
# ╟─6ed223f4-de31-43f7-a16f-16523c1d61ea
# ╠═8b2bd08c-866a-4c6e-a2fc-261dc8c05f2a
# ╠═73607123-a784-483e-9241-772e5937d59d
# ╟─13945989-fb32-4027-a67b-e2a9a9254446
# ╠═2e926dc4-0f12-411e-85e2-5dcffdcc1266
# ╠═b7674b98-526c-40ee-bf83-a9d6e7be6e4f
# ╟─f5e789b2-a62e-4818-90c3-76f39ea11aaa
# ╟─efa7736c-22c0-410e-94da-1df315f22bbf
# ╠═9b351fbd-2820-4353-b3fa-ec0e6d07d861
# ╟─b4451d05-1ac5-4962-88d8-e59d9ca225ea
# ╟─58b7267d-491d-40f0-b4ba-27ed0c9cc855
# ╟─ac76b646-7c28-4384-9f04-5e4de5df154f
# ╟─effbea0f-e9af-469c-a792-10078da46b39
# ╠═83a14158-33d1-4f16-85e1-2726c8fbbdfc
# ╟─4b31dca2-0195-4899-8a3a-e9772fabf495
# ╟─79e0deab-1e36-4863-ad10-187ed8555c72
# ╠═66d385ba-9c6e-4378-b4e0-e54a4df346a5
# ╠═76b7002a-5df0-4d77-b03f-6b677cc50de4
# ╟─db799fa2-0e48-43ee-9ee1-80ff8d2e5de7
# ╟─eb3a6009-e181-443c-bb77-021e867030e4
# ╟─01a1fd52-ff6c-44c6-ab1a-d1c141a4d54e
# ╠═521f5ffa-2c22-44c5-8bdb-67410431ca2e
# ╠═a828548f-175b-4cf0-b0aa-d9eef0477f4d
# ╠═a1043a1c-5840-4175-aa4a-ef432c353073
# ╟─f9b35e98-347f-4ebd-a690-790c7b0e03d8
# ╟─b1a44835-58d7-462f-a0d4-85bc02d3fdc6
# ╟─91e520aa-97a1-40b2-8936-93c93a63011c
# ╟─39daeb26-66d6-4a05-979f-76666444c73b
# ╟─0e8ea002-6bc8-4684-a72a-f7d7062eecc0
# ╟─9a9b3942-72f2-4c9e-88a5-af927634468c
# ╟─1ff198ea-afd5-4acc-bb67-019051ff149b
# ╟─44ece9ce-f9f1-46f3-90c6-cb0502c92c67
# ╟─5fe95aa5-f670-4329-a933-240a8c074dea
# ╟─43d68541-84a5-4a63-9d8f-43783cc27ccc
# ╟─5c6d39b0-9942-4173-9455-39cb3c174873
# ╟─99468dd9-4b97-48e6-803b-489dc1cefdf8
# ╠═0a0e7b32-e1f4-4d5c-8ebc-b5d06b61e6df
# ╟─f6d87e32-419a-48be-8054-f54fb6e4cef3
# ╟─0fd29811-9e17-4c97-b9b7-ec9cc51b435f
# ╟─71726572-0341-4344-9a3f-410d3bbc430a
# ╟─7ccf487c-49a1-49e9-bc19-e2f4e8a7d331
# ╠═e6f45063-e553-42ab-8344-69ff78ab520f
# ╟─8bb55d7e-1817-4b81-8de6-ad31191d08e8
# ╟─90a47e0b-b911-4728-80b5-6ed74607833d
# ╟─5d79b8c1-beea-4ff9-9830-0f5e1c4ef29f
# ╟─418755cb-765f-4a8c-805d-ceac36c7706c
# ╟─4625c35f-cd2d-4883-838a-57276e83d241
# ╠═0ddf43c5-6ba0-4d22-80ad-6ca8cf92f69a
# ╟─00cb431b-dced-45fd-a191-12bd59a096f5
# ╟─a5bfd185-aa77-4e46-a6b6-d43c4785a7fa
# ╟─a7b6ecbd-1407-44dd-809e-33311970af12
# ╟─a96e6942-06ab-42d3-a7e5-9c431a676d15
# ╟─201ec4fd-01b1-49c4-a104-3d619ffb447b
# ╠═8b544491-b892-499f-8146-e7d1f02aaac1
# ╠═18fe69ea-2f4b-41d2-b44e-f395af273891
# ╟─b0769c65-9b86-496e-85fc-a8dc43c55576
