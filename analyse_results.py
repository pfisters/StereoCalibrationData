from absl import app, logging, flags
from absl.flags import FLAGS, argparse_flags
from src.LogAnalyser import LogAnalyser
from utils.plotting import plot_points, plot_extrinsics

FLAGS = flags.FLAGS
flags.DEFINE_string('log_directory', './logs/1/logs-20-12-27-16-11-02', 'log directory')

def main(argv):
    analyser = LogAnalyser(FLAGS.log_directory)

    gt, pt = analyser.load('points')
    errors_norm, errors = analyser.compare(gt, pt, iterations=40)
    gt_ex, rl_ex = analyser.load_extrinsics()
    plot_points(errors, stacked=True, title='Errors [mm]')
    plot_extrinsics(gt_ex, rl_ex)

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass