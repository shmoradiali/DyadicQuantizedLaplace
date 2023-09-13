import os
from argparse import ArgumentParser
import json
import numpy as np
from tqdm import tqdm
from privacy import DQLMechanism
from distributed_mean_estimation import scalar_dme_list, scalar_dme_unif_x


SAVE_DIR = "dql_points"

if __name__ == "__main__":
    parser = ArgumentParser("Calculate plot points for DQL.")

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
        "--encoding",
        default="elias_gamma",
        type=str,
        help="The encoding method to be used(elias_gamma/elias_delta/binary).",
    )

    args = parser.parse_args()

    k = args.num_clients
    eps_from = args.eps_from
    eps_to = args.eps_to
    eps_subdiv = args.eps_subdiv
    encoding = args.encoding

    print("Calculating points for DQL...")

    filename = f"DQL_eps={eps_from:0.2f}_to_eps={eps_to:0.2f}_users={k}_enc={encoding}.json"

    xs = np.linspace(-1, 1, 50)
    xs_normalized = (xs + 1) / 2
    ls = list(np.linspace(1.1, 2.0, 22, endpoint=False)) + \
    list(np.linspace(2.0, 5.0, 15, endpoint=False)) + \
    list(np.linspace(5.0, 10.0, 15, endpoint=False)) + \
    list(np.linspace(5.0, 40.0, 20, endpoint=False)) + \
    list(np.linspace(40.0, 50.0, 18, endpoint=False))

    records = []

    for eps in tqdm(np.linspace(eps_from, eps_to, eps_subdiv)):
        for l in ls:
            enc = DQLMechanism(eps, l, shared_seed=42, local_seed=312)
            dec = DQLMechanism(eps, l, shared_seed=42, local_seed=694)

            avg_mse, avg_comm = scalar_dme_unif_x(k, enc, dec)

            # record: eps_server, eps_db, l, avg_comm, mse
            records.append({
                'eps_server': l * eps, 
                'eps_db': eps, 
                'l': l, 
                'avg_comm': avg_comm, 
                'avg_mse': avg_mse,
            })
    
    with open(os.path.join(SAVE_DIR, filename), 'w') as file:
        json.dump(records, file, indent=4)
