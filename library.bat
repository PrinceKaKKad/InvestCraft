@echo off
setlocal

REM Define required library versions
set "dash_version=2.15.0"
set "pandas_version=2.2.0"
set "plotly_version=5.18.0"
set "dash_daq_version=0.4.0"
set "cvxopt_version=1.3.2"
set "requests_version=2.31.0"
set "sklearn_version=1.3.2"
set "openai_version=0.28.0"
set "langchain_version=0.1.11"

echo We are installing the libraries please wait... it may take a while. Meanwhile, you can grab a coffee. :)

REM Check and install dash
python -c "import dash" 2>nul
if errorlevel 1 (
    echo Installing dash...
    pip install dash==%dash_version%
) else (
    python -c "import dash; print(dash.__version__)" | findstr /i "%dash_version%" >nul
    if errorlevel 1 (
        echo Reinstalling dash...
        pip install --force-reinstall dash==%dash_version%
    ) else (
        echo Dash is already installed.
    )
)

REM Check and install pandas
python -c "import pandas" 2>nul
if errorlevel 1 (
    echo Installing pandas...
    pip install pandas==%pandas_version%
) else (
    python -c "import pandas; print(pandas.__version__)" | findstr /i "%pandas_version%" >nul
    if errorlevel 1 (
        echo Reinstalling pandas...
        pip install --force-reinstall pandas==%pandas_version%
    ) else (
        echo Pandas is already installed.
    )
)

REM Check and install plotly
python -c "import plotly" 2>nul
if errorlevel 1 (
    echo Installing plotly...
    pip install plotly==%plotly_version%
) else (
    python -c "import plotly; print(plotly.__version__)" | findstr /i "%plotly_version%" >nul
    if errorlevel 1 (
        echo Reinstalling plotly...
        pip install --force-reinstall plotly==%plotly_version%
    ) else (
        echo Plotly is already installed.
    )
)

REM Check and install dash_daq
python -c "import dash_daq" 2>nul
if errorlevel 1 (
    echo Installing dash_daq...
    pip install dash-daq==%dash_daq_version%
) else (
    python -c "import dash_daq; print(dash_daq.__version__)" | findstr /i "%dash_daq_version%" >nul
    if errorlevel 1 (
        echo Reinstalling dash_daq...
        pip install --force-reinstall dash-daq==%dash_daq_version%
    ) else (
        echo Dash_daq is already installed.
    )
)

REM Check and install cvxopt
python -c "import cvxopt" 2>nul
if errorlevel 1 (
    echo Installing cvxopt...
    pip install cvxopt==%cvxopt_version%
) else (
    python -c "import cvxopt; print(cvxopt.__version__)" | findstr /i "%cvxopt_version%" >nul
    if errorlevel 1 (
        echo Reinstalling cvxopt...
        pip install --force-reinstall cvxopt==%cvxopt_version%
    ) else (
        echo Cvxopt is already installed.
    )
)

REM Check and install requests
python -c "import requests" 2>nul
if errorlevel 1 (
    echo Installing requests...
    pip install requests==%requests_version%
) else (
    python -c "import requests; print(requests.__version__)" | findstr /i "%requests_version%" >nul
    if errorlevel 1 (
        echo Reinstalling requests...
        pip install --force-reinstall requests==%requests_version%
    ) else (
        echo Requests is already installed.
    )
)

REM Check and install sklearn
python -c "import sklearn" 2>nul
if errorlevel 1 (
    echo Installing scikit-learn...
    pip install scikit-learn==%sklearn_version%
) else (
    python -c "import sklearn; print(sklearn.__version__)" | findstr /i "%sklearn_version%" >nul
    if errorlevel 1 (
        echo Reinstalling scikit-learn...
        pip install --force-reinstall scikit-learn==%sklearn_version%
    ) else (
        echo Scikit-learn is already installed.
    )
)

REM Check and install openai
python -c "import openai" 2>nul
if errorlevel 1 (
    echo Installing openai...
    pip install openai==%openai_version%
) else (
    python -c "import openai; print(openai.__version__)" | findstr /i "%openai_version%" >nul
    if errorlevel 1 (
        echo Reinstalling openai...
        pip install --force-reinstall openai==%openai_version%
    ) else (
        echo OpenAI is already installed.
    )
)

REM Check and install langchain
python -c "import langchain" 2>nul
if errorlevel 1 (
    echo Installing langchain...
    pip install langchain==%langchain_version%
) else (
    python -c "import langchain; print(langchain.__version__)" | findstr /i "%langchain_version%" >nul
    if errorlevel 1 (
        echo Reinstalling langchain...
        pip install --force-reinstall langchain==%langchain_version%
    ) else (
        echo Langchain is already installed.
    )
)

REM End of script
echo All libraries checked and installed/updated.
pause
