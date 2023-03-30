def check_is_running_in_colab():
    """Checks whether code is executed in google colab by
    attempting to import a library that is always available in
    colab.

    When running locally, this package should not be installed.
    Best use a python virtual environment with only the packages
    from the requirements.txt

    Returns
    -------
    bool
    """
    try:
        import google.colab
        return True
    except:
        return False
