from absl import app, logging, flags
from absl.flags import FLAGS, argparse_flags
from src.LogAnalyser import LogAnalyser
from utils.plotting import line_plot_points, plot_extrinsics

FLAGS = flags.FLAGS
flags.DEFINE_string('log_directory', './logs/logs-21-01-04-19-46-30', 'log directory')

def main(argv):
    analyser = LogAnalyser(FLAGS.log_directory, synthetic=False)

    # gt, pt = analyser.load('points')
    pt = analyser.load('points')
    # errors_norm, errors = analyser.compare(gt, pt, iterations=40)
    # gt_ex, rl_ex = analyser.load_extrinsics()
    rl_ex = analyser.load_extrinsics()
    # line_plot_points(errors, stacked=True, title='Errors [mm]')
    plot_extrinsics(rl_ex, iterations=60, title='Extrinsic Parameters')
    rms = analyser.load_extrinsics('rms', items=60)
    line_plot_points(rms, title='RMS [mm]')


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass