Command-line tools
==================

Installing the package provides four console scripts.

``starfit``
-----------

Fit a stellar model conditioned on the observations in a ``star.ini`` file
(format: see :doc:`examples`).

.. code-block:: console

    $ starfit                   # fit ./star.ini
    $ starfit mystar            # fit mystar/star.ini
    $ starfit star1 star2 -v    # several folders, verbose
    $ starfit --binary mystar   # also: --triple, or --all for single+binary+triple

``folders``
    folders containing ``star.ini`` (default ``.``)
``--binary``, ``--triple``, ``--all``
    fit a binary model, a triple model, or single + binary + triple
``--models``
    model grid name (default ``mist``)
``--emcee``
    sample with emcee instead of MultiNest
``--fehprior``
    metallicity prior (default ``local``)
``--plot_only``
    re-make the plots from an existing fit
``-o``, ``--overwrite``
    overwrite existing output
``-v``, ``--verbose``
    verbose logging
``--bands``
    additional band(s) to include in the samples

If ``pymultinest`` cannot be imported, ``--emcee`` is turned on
automatically.  Each fit writes an HDF5 file (for example
``mist_starmodel_single.h5``) plus corner plots into the folder.

``starfit-summarize``
---------------------

Collect posterior summaries of many fits into one table.

.. code-block:: console

    $ starfit-summarize star1 star2 star3 -o summary.h5
    $ starfit-summarize folder/* --rootdir results --processes 8

``folders``
    star folders to summarise
``--binary``, ``--triple``, ``--all``
    summarise the binary / triple / all model files
``--modelname``
    model file base name (default ``mist_starmodel_single``)
``-o``, ``--outfile``
    output file (default ``summary.h5``)
``--rootdir``
    directory containing the star folders
``-p``, ``--processes``
    worker processes
``--raise_exceptions``
    stop at the first unreadable model

The table contains, for each derived parameter, the quantiles
(``05, 16, 50, 84, 95``), the posterior mode, and the 95% central and minimum
credible intervals (``cci_lo_95``, ``mci_hi_95``, ...), as produced by
:func:`isochrones.summary.get_quantiles`.

``batch-starfit``
-----------------

Launch ``starfit`` over every folder listed in a text file, as a slurm job
array.

.. code-block:: console

    $ batch-starfit folders.txt --nsplit 20 --ntasks_per_node 20 --time 5 --binary

Arguments that ``batch-starfit`` does not recognise after the file name are
passed through to ``starfit``.  Adapt the script for another scheduler.

``starmodel-select``
--------------------

Model selection between different MultiNest runs of the same star.

.. code-block:: console

    $ starmodel-select star1 star2 --models mist
