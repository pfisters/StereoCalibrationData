from absl import app, logging, flags
from absl.flags import FLAGS, argparse_flags
from src.LogAnalyser import LogAnalyser
from utils.plotting import line_plot_points, plot_extrinsics

FLAGS = flags.FLAGS
flags.DEFINE_string('log_directory', './logs/4x4/std0', 'log directory')

def main(argv):

    synthetic = True
    analyser = LogAnalyser(FLAGS.log_directory, synthetic=synthetic)

    if synthetic: 
        gt, pt = analyser.load('points')
        errors_norm, errors = analyser.compare(gt, pt, iterations=-1)
        gt_ex, rl_ex = analyser.load_extrinsics()
        line_plot_points(errors, stacked=True, title='Errors [mm]')
        plot_extrinsics(rl_ex, ground_truth=gt_ex, title='Extrinsic Parameters')
        rms_gt, rms_rl = analyser.load_extrinsics('rms')
    else:
        pt = analyser.load('points')
        rl_ex = analyser.load_extrinsics()
        plot_extrinsics(rl_ex, title='Extrinsic Parameters')
        rms_rl = analyser.load_extrinsics('rms')
        line_plot_points(rms_rl, title='RMS [pixels]')


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass