# -*- coding: utf-8 -*-
from Markowitz.Markowitz import Markowitz


def main():
    mk = Markowitz(r'G:\PycharmProjects\EfficiencyFrontier\Example\GPM-2015-2024\*.tif')
    # mk.load_stack(
    #     block_size=716,
    #     threshold=0.0,
    #     pixel_presence=0.99,
    #     save_as=None,
    #     memmap_path='stack_float32.dat',
    #     dtype='float32'
    # )
    mk.load_datstack(memmap_path='stack_float32.dat', memmap_shape_path='stack_shape.json',
                     stack_metadata='stack_metadata.npz', dtype='float32')
    mk.calculate_statistics(
        num_pixels= 5,
        method= 'ledoitwolf',
        shrinkage_intensity= 0.3,
        normalize= True,
        norm_method= 'standard',
        sampling_method= 'kmeans_only',
        bayesian_grouping= 5,
        n_components= 5,
        norm_axis=0
    )
    mk.simulate_portfolios(num_portfolios=1000)
    mk.plot_sampled_space(use_hexbin=True)
    mk.plot_frontier(optimize=True)
    # sel, bn = mk.get_high_sharpe(.7)
    # mk.create_tif_from_array('output_mask.tif', bn)


if __name__ == "__main__":
    main()