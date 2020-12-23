from absl import app, logging, flags
from absl.flags import FLAGS, argparse_flags
from src.LogAnalyser import LogAnalyser
from utils.plotting import plot_points

FLAGS = flags.FLAGS
flags.DEFINE_string('log_directory', './logs/logs-20-12-19-17-28-28', 'log directory')

def main(argv):
    analyser = LogAnalyser(FLAGS.log_directory)

    gt, pt = analyser.load('points')
    errors = analyser.compare(gt, pt)
    gt_ex, rl_ex = analyser.load_extrinsics('rx')
    plot_points(gt_ex, rl_ex)

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass