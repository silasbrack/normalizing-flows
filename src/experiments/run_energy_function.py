from src.problems import energy_function
from src.guides import normalizing_flow
from src.visualization.visualize_distribution import plot_pdf_contours, plot_pdf_image
from pyro.distributions.transforms import planar, radial
from src.experiments import train

"""
We want to recreate figure 3 from Rezende et al, 2015.
"""

index = 0
for dist_name in (["U1", "U2", "U3", "U4"]):
    ax = plt.subplot(5, 5, index)
    plot_pdf_image(posterior.log_prob, ax, name=dist_name)
    index += 2

    for flow_type in ([planar, radial]):
        for n_flows in enumerate([2, 8, 32]):
            dist = get_distribution(dist_name)
            model = energy_function.model
            guide = normalizing_flow(model, num_flows=n_flows, flow_type=flow_type)

            train_result = train(model, guide, 20000, adam_params={"lr": 5e-3})

            posterior = guide.get_posterior()

            ax = plt.subplot(5, 9, index)
            plot_pdf_image(posterior.log_prob, ax, name=dist_name)

            index += 1
        index += 1
