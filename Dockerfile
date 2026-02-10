# syntax=docker/dockerfile:1

FROM rust:1.76-bullseye AS builder
WORKDIR /app

# Cache dependencies
COPY Cargo.toml Cargo.lock ./
COPY enkaic/Cargo.toml enkaic/
COPY enkai/Cargo.toml enkai/
COPY enkai_native/Cargo.toml enkai_native/
COPY enkai_tensor/Cargo.toml enkai_tensor/
COPY enkairt/Cargo.toml enkairt/
COPY std/std/Cargo.toml std/std/
COPY examples/hello/Cargo.toml examples/hello/ || true

RUN mkdir -p src && echo "fn main(){}" > src/main.rs
RUN cargo build --release -p enkai >/dev/null 2>&1 || true

# Build
COPY . .
RUN cargo build --release -p enkai

FROM debian:bullseye-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
ENV ENKAI_STD_PATH=/opt/enkai/std
COPY --from=builder /app/target/release/enkai /usr/local/bin/enkai
COPY std /opt/enkai/std
ENTRYPOINT ["enkai"]
CMD ["--help"]
