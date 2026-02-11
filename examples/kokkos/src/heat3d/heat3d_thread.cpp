#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <pthread.h>
#include <stdio.h>
#include <thread>
#include <utility>
#include <vector>

inline void cudaCheck(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<class ViewType>
int device_id_of_view(const ViewType& v)
{
  static_assert(std::is_same_v<typename ViewType::memory_space,
                               Kokkos::CudaSpace> ||
                std::is_same_v<typename ViewType::memory_space,
                               Kokkos::CudaUVMSpace>,
                "Works only for CUDA‐resident views");

  void* raw = const_cast<typename ViewType::value_type*>(v.data());
  cudaPointerAttributes attr{};
  cudaError_t err = cudaPointerGetAttributes(&attr, raw);
  if (err != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(err));   // pointer must be valid

  return attr.device;
}


enum { left, right, down, up, front, back, bulk };

struct Params {
  int X{200}, Y{200}, Z{200};
  int N{10000};
  int I{9999};
  double T0{0.0}, dt{0.1}, q{1.0}, sigma{1.0};
  volatile double P{1.0};
  int num_dev{1}; // number of devices

  bool parse(int argc, char **argv) {
    for (int i = 1; i < argc; ++i)
      if (!strcmp(argv[i], "-h"))
        return false;
    for (int i = 1; i < argc; ++i) {
      auto eq = [&](const char *s) { return strcmp(argv[i], s) == 0; };
      if (eq("-X"))
        X = atoi(argv[++i]);
      else if (eq("-Y"))
        Y = atoi(argv[++i]);
      else if (eq("-Z"))
        Z = atoi(argv[++i]);
      else if (eq("-N"))
        N = atoi(argv[++i]);
      else if (eq("-I"))
        I = atoi(argv[++i]);
      else if (eq("-T0"))
        T0 = atof(argv[++i]);
      else if (eq("-dt"))
        dt = atof(argv[++i]);
      else if (eq("-q"))
        q = atof(argv[++i]);
      else if (eq("-sigma"))
        sigma = atof(argv[++i]);
      else if (eq("-P"))
        P = atof(argv[++i]);
      else if (eq("-num_dev"))
        num_dev = atoi(argv[++i]);
    }
    return true;
  }
};

struct DeviceGrid {
  int nx{}, ny{}, nz{};                               // dims of processor grid
  int me{};                                           // this device index
  int num_dev{};                                      // total devices
  int x{}, y{}, z{};                                  // coords in grid
  int left{}, right{}, down{}, up{}, front{}, back{}; // neighbour ids or -1

  private:
    void locate(int device_id) {
      me = device_id;
      x = me % nx;
      y = (me / nx) % ny;
      z = me / (nx * ny);

      left = (x == 0 ? -1 : me - 1);
      right = (x == nx - 1 ? -1 : me + 1);
      down = (y == 0 ? -1 : me - nx);
      up = (y == ny - 1 ? -1 : me + nx);
      front = (z == 0 ? -1 : me - nx * ny);
      back = (z == nz - 1 ? -1 : me + nx * ny);
    }

  public:

    explicit DeviceGrid(int num_devices, int me) : num_dev(num_devices), me(me) {
      nx = std::pow(1.0 * num_devices, 1.0 / 3.0);
      while (num_devices % nx)
        ++nx;
      ny = std::sqrt(1.0 * (num_devices / nx));
      while ((num_devices / nx) % ny)
        ++ny;
      nz = num_devices / (nx * ny);

      locate(me);
    }
};

struct ExecOnDevice {
  Kokkos::Cuda exec;   
  cudaStream_t stream;
  std::array<Kokkos::Cuda, 7> partitioned_exec;
  int dev_id;
  std::vector<std::reference_wrapper<ExecOnDevice>> neighbor_spaces;

  explicit ExecOnDevice(int dev) : dev_id(dev) {
    cudaCheck(cudaSetDevice(dev), "cudaSetDevice");
    cudaCheck(cudaStreamCreate(&stream), "cudaStreamCreate");
    exec = Kokkos::Cuda(stream);
    // std::cout << "ExecOnDevice created for device " << dev_id << std::endl;
    // std::cout << "Stream: " << stream << std::endl;

    for (int i = 0; i < 7; ++i) {
      partitioned_exec[i] = Kokkos::Cuda(stream);
    }
  }
  ~ExecOnDevice() {
    cudaCheck(cudaSetDevice(dev_id), "cudaSetDevice");
    cudaCheck(cudaStreamDestroy(stream), "cudaStreamDestroy");
  }
  ExecOnDevice(const ExecOnDevice &) = delete;
  ExecOnDevice &operator=(const ExecOnDevice &) = delete;
  ExecOnDevice(ExecOnDevice &&) = default;
  ExecOnDevice &operator=(ExecOnDevice &&) = default;
};

struct DeviceField {
  using view3d_t =
      Kokkos::View<double ***, Kokkos::CudaSpace>;
  using buffer_t =
      Kokkos::View<double **, Kokkos::CudaSpace>;

  int X_lo{}, X_hi{}, Y_lo{}, Y_hi{}, Z_lo{}, Z_hi{}; // boundaries
  int NX{}, NY{}, NZ{}; // dimensions of the local field
  int X{}, Y{}, Z{};    // global dimensions

  double P{};
  double dt{};
  double T0{};
  double q{};
  double sigma{};


  view3d_t T, dT;
  // six 2‑D halo planes  (nullptr if boundary)
  buffer_t T_left, T_right, T_down, T_up, T_front, T_back;
  buffer_t T_left_out, T_right_out, T_down_out, T_up_out, T_front_out, T_back_out;

  struct ComputeInnerDT {};

  KOKKOS_FUNCTION
  void operator()(ComputeInnerDT, int x, int y, int z) const {
    double dT_xyz = 0.0;
    double T_xyz = T(x, y, z);
    dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    dT_xyz += q * (T(x, y, z + 1) - T_xyz);
    dT(x, y, z) = dT_xyz;
  }

  void compute_inner(ExecOnDevice& space) {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, ComputeInnerDT, int>;
    int myX = T.extent(0);
    int myY = T.extent(1);
    int myZ = T.extent(2);

    // std::cout << "T : " << device_id_of_view(T) << std::endl;

    Kokkos::parallel_for(
        "ComputeInnerDT",
        Kokkos::Experimental::require(
            policy_t(space.partitioned_exec[bulk], {1, 1, 1},
                     {myX - 1, myY - 1, myZ - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
  };

  template <int Surface> struct ComputeSurfaceDT {};

  template <int Surface>
  KOKKOS_FUNCTION void operator()(ComputeSurfaceDT<Surface>, int i,
                                  int j) const {
    int NX = T.extent(0);
    int NY = T.extent(1);
    int NZ = T.extent(2);
    int x, y, z;
    if (Surface == left) {
      x = 0;
      y = i;
      z = j;
    }
    if (Surface == right) {
      x = NX - 1;
      y = i;
      z = j;
    }
    if (Surface == down) {
      x = i;
      y = 0;
      z = j;
    }
    if (Surface == up) {
      x = i;
      y = NY - 1;
      z = j;
    }
    if (Surface == front) {
      x = i;
      y = j;
      z = 0;
    }
    if (Surface == back) {
      x = i;
      y = j;
      z = NZ - 1;
    }

    double dT_xyz = 0.0;
    double T_xyz = T(x, y, z);

    // Heat conduction to inner body
    if (x > 0)
      dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    if (x < NX - 1)
      dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    if (y > 0)
      dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    if (y < NY - 1)
      dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    if (z > 0)
      dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    if (z < NZ - 1)
      dT_xyz += q * (T(x, y, z + 1) - T_xyz);

    // Heat conduction with Halo
    if (x == 0 && X_lo != 0)
      dT_xyz += q * (T_left(y, z) - T_xyz);
    if (x == (NX - 1) && X_hi != X)
      dT_xyz += q * (T_right(y, z) - T_xyz);
    if (y == 0 && Y_lo != 0)
      dT_xyz += q * (T_down(x, z) - T_xyz);
    if (y == (NY - 1) && Y_hi != Y)
      dT_xyz += q * (T_up(x, z) - T_xyz);
    if (z == 0 && Z_lo != 0)
      dT_xyz += q * (T_front(x, y) - T_xyz);
    if (z == (NZ - 1) && Z_hi != Z)
      dT_xyz += q * (T_back(x, y) - T_xyz);

    // Incoming Power
    if (x == 0 && X_lo == 0)
      dT_xyz += P;

    // Thermal radiation
    int num_surfaces = ((x == 0 && X_lo == 0) ? 1 : 0) +
                       ((x == (NX - 1) && X_hi == X) ? 1 : 0) +
                       ((y == 0 && Y_lo == 0) ? 1 : 0) +
                       ((y == (NY - 1) && Y_hi == Y) ? 1 : 0) +
                       ((z == 0 && Z_lo == 0) ? 1 : 0) +
                       ((z == (NZ - 1) && Z_hi == Z) ? 1 : 0);
    dT_xyz -= sigma * T_xyz * T_xyz * T_xyz * T_xyz * num_surfaces;
    dT(x, y, z) = dT_xyz;
  }

  void compute_surface(ExecOnDevice& space) {
    using policy_left_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<left>, int>;
    using policy_right_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<right>, int>;
    using policy_down_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<down>, int>;
    using policy_up_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<up>, int>;
    using policy_front_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<front>, int>;
    using policy_back_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<back>, int>;

    int X = T.extent(0);
    int Y = T.extent(1);
    int Z = T.extent(2);

    Kokkos::parallel_for(
        "ComputeSurfaceDT_Left",
        Kokkos::Experimental::require(
            policy_left_t(space.partitioned_exec[left], {0, 0}, {Y, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Right",
        Kokkos::Experimental::require(
            policy_right_t(space.partitioned_exec[right], {0, 0}, {Y, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Down",
        Kokkos::Experimental::require(
            policy_down_t(space.partitioned_exec[down], {1, 0}, {X - 1, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Up",
        Kokkos::Experimental::require(
            policy_up_t(space.partitioned_exec[up], {1, 0}, {X - 1, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_front",
        Kokkos::Experimental::require(
            policy_front_t(space.partitioned_exec[front], {1, 1}, {X - 1, Y - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::parallel_for(
        "ComputeSurfaceDT_back",
        Kokkos::Experimental::require(
            policy_back_t(space.partitioned_exec[back], {1, 1}, {X - 1, Y - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
  }

  struct UpdateT {
    view3d_t T, dT;
    double dt;
    UpdateT(view3d_t T_, view3d_t dT_, double dt_)
        : T(T_), dT(dT_), dt(dt_) {}
    KOKKOS_FUNCTION
    void operator()(int x, int y, int z, double &sum_T) const {
      sum_T += T(x, y, z);
      T(x, y, z) += dt * dT(x, y, z);
    }
  };

  double update_t(ExecOnDevice& space) {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::IndexType<int>>;
    int X = T.extent(0);
    int Y = T.extent(1);
    int Z = T.extent(2);
    double my_T = 0.0;
    Kokkos::parallel_reduce(
        "UpdateT",
        Kokkos::Experimental::require(
            policy_t(space.partitioned_exec[bulk], {0, 0, 0}, {X, Y, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        UpdateT(T, dT, dt), my_T);
    return my_T;
  }
};


class System {
  public:
    std::vector<double> local_T_average;
    volatile double power = 0.0;

    Params params;
    int num_dev;
    int num_real_dev{4};
    std::vector<ExecOnDevice> exec_spaces;
    std::vector<DeviceGrid> grids;
    std::vector<DeviceField> fields;

    void build_subdomain(int dev) {

      cudaSetDevice(dev);
      auto &grid = grids[dev];
      auto &data = fields[dev];

      // Domain partition (based on MPI script)
      const int dX = (params.X + grid.nx - 1) / grid.nx;
      const int dY = (params.Y + grid.ny - 1) / grid.ny;
      const int dZ = (params.Z + grid.nz - 1) / grid.nz;

      const int X_lo = dX * grid.x;
      const int Y_lo = dY * grid.y;
      const int Z_lo = dZ * grid.z;

      const int X_hi = std::min(params.X, X_lo + dX);
      const int Y_hi = std::min(params.Y, Y_lo + dY);
      const int Z_hi = std::min(params.Z, Z_lo + dZ);

      const int NX = X_hi - X_lo;
      const int NY = Y_hi - Y_lo;
      const int NZ = Z_hi - Z_lo;

      data.P = params.P;
      data.dt = params.dt;
      data.T0 = params.T0;
      data.q = params.q;
      data.sigma = params.sigma;  
      data.X = params.X;
      data.Y = params.Y;
      data.Z = params.Z;


      data.X_lo = X_lo;
      data.X_hi = X_hi;
      data.Y_lo = Y_lo;
      data.Y_hi = Y_hi;
      data.Z_lo = Z_lo;
      data.Z_hi = Z_hi;
      data.NX = NX;
      data.NY = NY;
      data.NZ = NZ;

      auto& space = exec_spaces[dev];
      // Allocate core arrays
      data.T =
          DeviceField::view3d_t(Kokkos::view_alloc(space.exec), NX, NY, NZ);
      data.dT =
          DeviceField::view3d_t(Kokkos::view_alloc(space.exec), NX, NY, NZ);
      Kokkos::deep_copy(space.exec, data.T, params.T0);

      auto make_plane = [&](const char *name, int U, int V) {
        return DeviceField::buffer_t(
            Kokkos::view_alloc(space.exec, "name"), U,
            V);
      };

      // Allocate halo buffers where neighbour exists
      if (data.X_lo != 0) {
        data.T_left = make_plane("left", NY, NZ);
        data.T_left_out = make_plane("left_o", NY, NZ);
      }
      if (data.X_hi != params.X) {
        data.T_right = make_plane("right", NY, NZ);
        data.T_right_out = make_plane("right_o", NY, NZ);
      }
      if (data.Y_lo != 0) {
        data.T_down = make_plane("down", NX, NZ);
        data.T_down_out = make_plane("down_o", NX, NZ);
      }
      if (data.Y_hi != params.Y) {
        data.T_up = make_plane("up", NX, NZ);
        data.T_up_out = make_plane("up_o", NX, NZ);
      }
      if (data.Z_lo != 0) {
        data.T_front = make_plane("front", NX, NY);
        data.T_front_out = make_plane("front_o", NX, NY);
      }
      if (data.Z_hi != params.Z) {
        data.T_back = make_plane("back", NX, NY);
        data.T_back_out = make_plane("back_o", NX, NY);
      }
  }

  explicit System(const Params &p, int num_real_dev = 4)
      : params(p), num_real_dev(num_real_dev) {
    num_dev = params.num_dev;
    power = params.P;


    auto id_to_device = [&](int id){
      return id % num_real_dev;
    };

    grids.reserve(num_dev);
    fields.resize(num_dev);
    exec_spaces.reserve(num_dev);
    local_T_average.resize(num_dev, 0.0);

    for (int dev = 0; dev < num_dev; ++dev) {
      grids.emplace_back(num_dev, dev);
      exec_spaces.emplace_back(id_to_device(dev));
    }
  }

  void build_domain(int dev) {
    build_subdomain(dev);
    initialize_neighbors(dev);
  }

  void pack_halos(int dev){

    auto &grid = grids[dev];
    auto &data = fields[dev];
    auto &space = exec_spaces[dev];

    if (grid.left != -1){
      // std::cout << "Packing left halo for device " << dev << std::endl;
      Kokkos::deep_copy(space.partitioned_exec[left], data.T_left_out,
                        Kokkos::subview(data.T, 0, Kokkos::ALL, Kokkos::ALL));
      }

    if (grid.right != -1){
      // std::cout << "Packing right halo for device " << dev << std::endl;
      Kokkos::deep_copy(
          space.partitioned_exec[right], data.T_right_out,
          Kokkos::subview(data.T, data.T.extent(0) - 1, Kokkos::ALL, Kokkos::ALL));
      }

    if (grid.down != -1){
      // std::cout << "Packing down halo for device " << dev << std::endl;
      Kokkos::deep_copy(space.partitioned_exec[down], data.T_down_out,
                        Kokkos::subview(data.T, Kokkos::ALL, 0, Kokkos::ALL));
    }

    if (grid.up != -1){
      // std::cout << "Packing up halo for device " << dev << std::endl;
      Kokkos::deep_copy(
          space.partitioned_exec[up], data.T_up_out,
          Kokkos::subview(data.T, Kokkos::ALL, data.T.extent(1) - 1, Kokkos::ALL));
    }

    if (grid.front != -1){
      // std::cout << "Packing front halo for device " << dev << std::endl;
      Kokkos::deep_copy(space.partitioned_exec[front], data.T_front_out,
                        Kokkos::subview(data.T, Kokkos::ALL, Kokkos::ALL, 0));
    }

    if (grid.back != -1){
      // std::cout << "Packing back halo for device " << dev << std::endl;
      Kokkos::deep_copy(
          space.partitioned_exec[back], data.T_back_out,
          Kokkos::subview(data.T, Kokkos::ALL, Kokkos::ALL,
                          data.T.extent(2) - 1));
    }
  }

  void compute_inner(int dev) {
    auto& f = fields[dev];
    auto& space = exec_spaces[dev];
    f.compute_inner(space);
  }

  void initialize_neighbors(int dev){
    auto &space = exec_spaces[dev];
    auto &grid = grids[dev];

    if (grid.left != -1) space.neighbor_spaces.push_back(exec_spaces[grid.left]);
    if (grid.right != -1)
      space.neighbor_spaces.push_back(std::ref(exec_spaces[grid.right]));
    if (grid.down != -1)
      space.neighbor_spaces.push_back(std::ref(exec_spaces[grid.down]));
    if (grid.up != -1)
      space.neighbor_spaces.push_back(std::ref(exec_spaces[grid.up]));
    if (grid.front != -1)
      space.neighbor_spaces.push_back(std::ref(exec_spaces[grid.front]));
    if (grid.back != -1)
      space.neighbor_spaces.push_back(std::ref(exec_spaces[grid.back]));
  }

  void synchronize_neighbors(int dev) {
    auto &space = exec_spaces[dev];
    auto n_neighbors = space.neighbor_spaces.size();
    for (int i = 0; i < n_neighbors; ++i) {
      space.neighbor_spaces[i].get().exec.fence();
    }
  }

  void synchronize_everything(){
    for(int dev = 0; dev < num_dev; ++dev) {
      synchronize_neighbors(dev);
      synchronize_self(dev);
    }
    Kokkos::fence();
  }

  void exchange_halos(int dev) {
    auto &space = exec_spaces[dev];
    auto &grid = grids[dev];
    auto &f = fields[dev];
    auto X_lo = f.X_lo, X_hi = f.X_hi;
    auto Y_lo = f.Y_lo, Y_hi = f.Y_hi;
    auto Z_lo = f.Z_lo, Z_hi = f.Z_hi;
    auto X = params.X, Y = params.Y, Z = params.Z;

    // left ↔ right  (device pair)
    if (X_lo != 0) {
      //I have a left neighbor (and am the right neighbor of someone)
      //Push my left halo to my neighbors right buffer
      //Data is sent on my neighbors stream
      space.partitioned_exec[left].fence();
      auto &f_nei = fields[grid.left];
      // std::cout << "Sending left halo from device " << dev << " to device "
      //           << grid.left << std::endl;
      Kokkos::deep_copy(space.exec, f_nei.T_right, f.T_left_out);
    }
    if (X_hi != X) {
      //I have a right neighbor (and am the left neighbor of someone)
      //Push my right halo to my neighbors left buffer
      //Data is sent on my neighbors stream
      space.partitioned_exec[right].fence();
      auto &f_nei = fields[grid.right];
      // std::cout << "Sending right halo from device " << dev << " to device "
      //           << grid.right << std::endl;
      Kokkos::deep_copy(space.exec, f_nei.T_left, f.T_right_out);
    }
    if (Y_lo != 0) {
      //I have a down neighbor (and am the up neighbor of someone)
      //Push my down halo to my neighbors up buffer
      //Data is sent on my neighbors stream
      space.partitioned_exec[down].fence();
      auto &f_nei = fields[grid.down];
      // std::cout << "Sending down halo from device " << dev << " to device "
      //           << grid.down << std::endl;
      Kokkos::deep_copy(space.exec, f_nei.T_up, f.T_down_out);
    }
    if (Y_hi != Y) {
      //I have an up neighbor (and am the down neighbor of someone)
      //Push my up halo to my neighbors down buffer
      //Data is sent on my neighbors stream
      space.partitioned_exec[up].fence();
      auto &f_nei = fields[grid.up];
      // std::cout << "Sending up halo from device " << dev << " to device "
      //           << grid.up << std::endl;
      Kokkos::deep_copy(space.exec, f_nei.T_down, f.T_up_out);
    }
    if (Z_lo != 0) {
      //I have a front neighbor (and am the back neighbor of someone)
      //Push my front halo to my neighbors back buffer
      //Data is sent on my neighbors stream
      space.partitioned_exec[front].fence();
      auto &f_nei = fields[grid.front];
      // std::cout << "Sending front halo from device " << dev << " to device "
      //           << grid.front << std::endl;
      Kokkos::deep_copy(space.exec, f_nei.T_back, f.T_front_out);
    }
    if (Z_hi != Z) {
      //I have a back neighbor (and am the front neighbor of someone)
      //Push my back halo to my neighbors front buffer
      //Data is sent on my neighbors stream
      space.partitioned_exec[back].fence();
      auto &f_nei = fields[grid.back];
      // std::cout << "Sending back halo from device " << dev << " to device "
      //           << grid.back << std::endl;
      Kokkos::deep_copy(space.exec, f_nei.T_front, f.T_back_out);
    }
  }

  void synchronize_self(int dev) {
    auto& space = exec_spaces[dev];
    for (auto &exec : space.partitioned_exec) {
      exec.fence();
    }
    space.exec.fence();
  }


  void compute_surface(int dev){
    auto &f = fields[dev];
    auto &space = exec_spaces[dev];
    f.compute_surface(space);
  }

  double update_T(int dev) {
    auto &f = fields[dev];
    auto &space = exec_spaces[dev];
    return f.update_t(space);
  }
};



void run_thread(int dev, System &sys, pthread_barrier_t &barrier) {

  cudaCheck(cudaSetDevice(sys.exec_spaces[dev].dev_id), "cudaSetDevice(thread)");

  for (int t = 0; t <= sys.params.N; ++t) {
    if (t > sys.params.N / 2) {
      sys.fields[dev].P = 0.0; 
    }

    sys.pack_halos(dev);
    sys.synchronize_self(dev);

    sys.compute_inner(dev);
    sys.synchronize_self(dev);

    sys.exchange_halos(dev);
    pthread_barrier_wait(&barrier);

    sys.synchronize_neighbors(dev);
    sys.compute_surface(dev);

    sys.synchronize_self(dev);

    double local_T_ave = 0;
    local_T_ave = sys.update_T(dev);
    sys.synchronize_self(dev);
    sys.local_T_average[dev] = local_T_ave;

    pthread_barrier_wait(&barrier);
    double T_ave = std::reduce(sys.local_T_average.begin(), sys.local_T_average.end());

    // if (dev == 0) {
    //   printf("Timestep: %i/%i t_avg: %lf\n", t, sys.params.N, T_ave);
    // }
  }

  if (dev == 0) {
    double T_ave = std::reduce(sys.local_T_average.begin(), sys.local_T_average.end());
    printf("Final average temperature: %lf\n", T_ave);
  }
}


int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Params params;
    if (!params.parse(argc, argv)) {
      printf("Usage: %s [-X <x>] [-Y <y>] [-Z <z>] [-N <steps>] "
             "[-I <interval>] [-T0 <initial T>] [-dt <time step>] "
             "[-q <heat transfer coefficient>] [-sigma <radiation coefficient>] "
             "[-P <power on X face>]\n",
             argv[0]);
      return 1;
    }

    const int num_real_devices = Kokkos::num_devices();
    std::cout << "Number of System Devices: " << num_real_devices << std::endl;
    std::cout << "Number of Logical Devices: " << params.num_dev << std::endl;

    System sys(params, num_real_devices);
    double time_all = 0.0;

    Kokkos::Timer timer;
    double time_start = timer.seconds();

    for (int dev = 0; dev < params.num_dev; ++dev) {
      sys.build_domain(dev);
    }
    sys.synchronize_everything();


    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, params.num_dev);

    std::vector<std::thread> t(params.num_dev);

    for (int dev = 0; dev < params.num_dev; ++dev) {
      t[dev] = std::thread(run_thread, dev, std::ref(sys), std::ref(barrier));
    }

    for (int dev = 0; dev < params.num_dev; ++dev) {
      t[dev].join();
    }

    double time_end = timer.seconds();
    time_all = time_end - time_start;
    std::cout << "Total time: " << time_all << " seconds" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}