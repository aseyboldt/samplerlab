from functools import partial

from samplerlab._model_registry import stan_model


@stan_model
def normal():
    code = """
        data {
            real mu;
        }
        parameters {
            real x;
        }
        model {
            x ~ normal(mu, 1);
        }
    """

    return code, {"mu": 1.0}


def register_posteriordb(path):
    import posteriordb

    db = posteriordb.PosteriorDatabase(path)
    names = db.posterior_names()

    def from_db(code, data):
        return code, data

    slow = [
        "ecdc0401-covid19imperial_v2",
        "ecdc0401-covid19imperial_v3",
        "ecdc0501-covid19imperial_v2",
        "ecdc0501-covid19imperial_v3",
        "election88-election88_full",
        "hmm_gaussian_simulated-hmm_gaussian",
        "iohmm_reg_simulated-iohmm_reg",
        "mnist-nn_rbm1bJ100",
        "mnist_100-nn_rbm1bJ10",
        "prideprejudice_chapter-ldaK5",
        "prideprejudice_paragraph-ldaK5",
    ]

    names.sort()

    for name in names:
        posterior = db.posterior(name)
        code = posterior.model.code("stan")
        data = posterior.data.values()
        keywords = posterior.data.information["keywords"]
        if keywords is None:
            keywords = []
        if isinstance(keywords, str):
            keywords = [keywords]

        if name not in slow:
            keywords.append("posteriordb-fast")

        keywords.append("posteriordb")

        stan_model(partial(from_db, code=code, data=data), name=name, keywords=keywords)
