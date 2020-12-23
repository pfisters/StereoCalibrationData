from absl import app, logging, flags
from absl.flags import FLAGS, argparse_flags
from src.LogAnalyser import LogAnalyser
from utils.plotting import plot_points

FLAGS = flags.FLAGS
flags.DEFINE_string('log_directory', './logs/logs-20-12-23-11-18-58', 'log directory')

def main(argv):
    analyser = LogAnalyser(FLAGS.log_directory)

    gt, pt = analyser.load('l_')
    errors = analyser.compare(gt, pt)
    gt_ex, rl_ex = analyser.load_extrinsics('tz')
    plot_points(gt_ex, rl_ex)

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass