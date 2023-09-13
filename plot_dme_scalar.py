import os
from argparse import ArgumentParser
import json
import numpy as np
import matplotlib.pyplot as plt


LOAD_DIR = "dql_points"

plot_line_width = 2.5
fig = plt.figure()
plt.rcParams.update({'font.size': 20})

if __name__ == "__main__":
    parser = ArgumentParser(description="Distributed Mean Estimation")
    parser.add_argument(
        "--comm_budget",
        default=5.0,
        type=float,
        help="Average communication budget to be used.",
    )
    parser.add_argument(
        "--num_clients",
        default=1000,
        type=int,
        help="Number of clients.",
    )
    parser.add_argument(
        "--eps_from",
        default=1,
        type=float,
        help="Minimum privacy budget.",
    )
    parser.add_argument(
        "--eps_to",
        default=20,
        type=float,
        help="Maximum privacy budget.",
    )
    parser.add_argument(
        "--eps_subdiv",
        default=40,
        type=int,
        help="Number of epsilons to use in [eps_from, eps_to].",
    )
    parser.add_argument(
        "--l_subdiv",
        default=60,
        type=int,
        help="Subdivision factor for the decoder privacy relaxation parameter.",
    )
    parser.add_argument(
        "--encoding",
        default="elias_gamma",
        type=str,
        help="The encoding method to be used(elias_gamma/elias_delta/binary).",
    )

    # Running DQL
    args = parser.parse_args()
    
    comm_budget_dql = args.comm_budget
    k = args.num_clients
    eps_from = args.eps_from
    eps_to = args.eps_to
    eps_subdiv = args.eps_subdiv
    l_subdiv = args.l_subdiv
    encoding = args.encoding

    print("Calculating points for DQL...")

    filename = f"DQL_eps={eps_from:0.2f}_to_eps={eps_to:0.2f}_users={k}_enc={encoding}.json"

    try:
        with open(os.path.join(LOAD_DIR, filename)) as file:
            points = json.load(file)
    except:
        print("First run preprocess_dql.py to calculate the plot points.")
        exit()
        
    min_l_for_eps = {}
    lte_budget_list = []
    for p in points:
        if p['avg_comm'] <= comm_budget_dql:
            min_l_for_eps[p['eps_db']] = min(p['l'], min_l_for_eps.get(p['eps_db'], 10**5))
            lte_budget_list.append(p)
    
    server_list, db_list = [], []
    for p in lte_budget_list:
        if p['l'] <= min_l_for_eps[p['eps_db']]:
            server_list.append((p['avg_mse'] , p['eps_server']))
            db_list.append((p['avg_mse'], p['eps_db'])) 
    
    # Plot server privacy
    server_list.sort()
    xs = [np.log10(mse) for mse, _ in server_list]
    ys = [np.log10(eps) for _, eps in server_list]
    plt.plot(xs, ys, linestyle=(0, (5, 1.8)), color='blue', linewidth=plot_line_width, label='DQL: decoder d-privacy')

    # Plot database privacy
    db_list.sort()
    xs = [np.log10(mse) for mse, _ in db_list]
    ys = [np.log10(eps) for _, eps in db_list]
    plt.plot(xs, ys, linestyle='-', color='blue', linewidth=plot_line_width, label='DQL: database d-privacy')

    print("Done.\n")

    # Show plot

    plt.xlabel("log10(MSE)")
    plt.ylabel("log10(epsilon)")

    plt.xlim(-1.62, 0)
    plt.ylim(0, 2.4)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    os.makedirs("plots", exist_ok=True)
    fig.savefig(os.path.join("plots", f"plot-eps={eps_from}_to_eps={eps_to}-b={comm_budget_dql}-clients={k}.pdf"), bbox_inches='tight')
