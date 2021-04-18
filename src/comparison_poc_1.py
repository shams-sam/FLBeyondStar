import argparse
import matplotlib
import matplotlib.pyplot as plt
from common.utils import Struct, history_parser


matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['lines.linewidth'] = 2.5
matplotlib.rcParams['lines.markersize'] = 7

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, required=False, default='mnist')
ap.add_argument('--num-nodes', type=int, required=False, default=125)
ap.add_argument('--epochs', type=int, required=False)
ap.add_argument('--histories', type=str, nargs='+', required=True)
ap.add_argument('--labels', type=str, nargs='+', required=True)
ap.add_argument('--name', type=str, required=True)
ap.add_argument('--ncols', type=int, required=True)
ap.add_argument('--dpi', type=int, required=True)
ap.add_argument('--colors', type=str, nargs='+', required=False, default=[])
ap.add_argument('--legend', type=int, required=False, default=1)
args = vars(ap.parse_args())
args = Struct(**args)

fig = plt.figure(figsize=(5, 4))
ax1 = fig.add_subplot(111)

colors = ['k.-', 'r.:', 'm.:', 'b.:', 'g.:', 'c.:', 'y.:', 'k.:', 'r', 'b']
if len(args.colors):
    colors = args.colors

hist1, hist2, hist3 = args.histories[:6], args.histories[6:12], args.histories[12:]

for idx, history in enumerate(hist2):
    x_ax, y_ax, l_test = history_parser(args.dataset, args.num_nodes, history)
    x_ax = x_ax[:args.epochs]
    y_ax = y_ax[:args.epochs]

    ax1.plot(x_ax, y_ax, colors[idx], label=args.labels[idx])

ax1.set_xlabel('t')
ax1.set_ylabel('accuracy')
ax1.grid()

if args.legend:
    ax1.legend(loc='upper right', ncol=args.ncols,
               bbox_to_anchor=(-0.25, 1.1, 1.32, .38),
               mode='expand', frameon=False, handlelength=1)


print('Saving: ', args.name)
fig.subplots_adjust(wspace=0.35, hspace=0.5)
plt.savefig('../ckpts/{}_{}/plots/{}'.format(
    args.dataset, args.num_nodes, args.name),
            bbox_inches='tight', dpi=args.dpi)
