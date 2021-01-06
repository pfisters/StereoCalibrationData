from absl import app, logging, flags
from absl.flags import FLAGS, argparse_flags
from src.LogAnalyser import LogAnalyser
from utils.plotting import line_plot_points, plot_extrinsics

FLAGS = flags.FLAGS
flags.DEFINE_string('log_directory', './logs/4x4_optim/std10/', 'log directory')

def main(argv):

    analyser = LogAnalyser(FLAGS.log_directory, synthetic=True)
    gt, pt = analyser.load('points')
    # pt = analyser.load('points')
    errors_norm, errors = analyser.compare(gt, pt, iterations=-1)
    gt_ex, rl_ex = analyser.load_extrinsics()
    # rl_ex = analyser.load_extrinsics()
    line_plot_points(errors, stacked=True, title='Errors [mm]')
    plot_extrinsics(rl_ex, ground_truth=gt_ex, title='Extrinsic Parameters')
    rms_gt, rms_rl = analyser.load_extrinsics('rms')
    line_plot_points(rms_rl, title='RMS [mm]')

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass