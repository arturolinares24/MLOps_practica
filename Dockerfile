FROM apache/airflow:2.9.0-python3.9
COPY requirements.txt /requirements.txt

# Establece las variables de entorno para la autenticación de Kaggle
ENV KAGGLE_USERNAME="arturolinares"
ENV KAGGLE_KEY="5b239d1dc000d187595039f836a68a40"

RUN pip install --upgrade pip

# Crea el directorio faltante y establece los permisos adecuados
USER root
RUN mkdir -p /var/lib/apt/lists/partial \
    && chmod 755 /var/lib/apt/lists/partial


# Actualiza paquetes e instala libgomp1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++ \
        git && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get install libgomp1 -y

USER airflow

RUN pip install --no-cache-dir -r /requirements.txt
