import click
import sys
from datalib import config
import yaml


def main(path):
    with open(path, 'r') as f:
        d = yaml.load(f)
    config.update(d)
    if config['DIST'] == 'SBD' and config['NO_SHIFT']:
        config.update(DIST='SBD_no_shift')

    from algorithm.cluster import Clusterer, Evaluator, draw_cluster_medoids, draw_each_cluster
    clu = Clusterer(min_samples=4, dist_measure=config['DIST'],
                 max_radius=config['MAX_RADIUS'], inflect_thresh=config['INFLECT_THRESH'], train_sim_matrix=config['RETRAIN'])
    y_pred, test_pred = clu.run()

    if config['EVALUATE']:
        eva = Evaluator(y_pred, test_pred, config['IGNORE'])
        eva.run()

    draw_each_cluster(config['MAX_RADIUS'], config['DRAW_START'], config['DRAW_END'])


if __name__ == '__main__':
    main(sys.argv[1])
