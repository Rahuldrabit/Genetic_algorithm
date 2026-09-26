#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "ga/fuzzy/fuzzy_types.hpp"

namespace ga {
namespace fuzzy {

class IMembershipFunction {
public:
    virtual ~IMembershipFunction() = default;
    virtual double evaluate(double x) const = 0;
    virtual Interval support() const = 0;
    virtual std::unique_ptr<IMembershipFunction> clone() const = 0;
};

class TriangularMF final : public IMembershipFunction {
public:
    TriangularMF(double a, double b, double c) : a_(a), b_(b), c_(c) {
        if (!(a_ <= b_ && b_ <= c_)) throw std::invalid_argument("TriangularMF: must have a <= b <= c");
    }
    double evaluate(double x) const override {
        if (x <= a_ || x >= c_) return 0.0;
        if (std::abs(x - b_) < 1e-12) return 1.0;
        return (x < b_) ? (x - a_) / (b_ - a_) : (c_ - x) / (c_ - b_);
    }
    Interval support() const override { return {a_, c_}; }
    std::unique_ptr<IMembershipFunction> clone() const override { return std::make_unique<TriangularMF>(*this); }
    double a() const noexcept { return a_; } double b() const noexcept { return b_; } double c() const noexcept { return c_; }
private:
    double a_, b_, c_;
};

class TrapezoidalMF final : public IMembershipFunction {
public:
    TrapezoidalMF(double a, double b, double c, double d) : a_(a), b_(b), c_(c), d_(d) {
        if (!(a_ <= b_ && b_ <= c_ && c_ <= d_)) throw std::invalid_argument("TrapezoidalMF: must have a <= b <= c <= d");
    }
    double evaluate(double x) const override {
        if (x <= a_ || x >= d_) return 0.0;
        if (x >= b_ && x <= c_) return 1.0;
        return (x < b_) ? (x - a_) / (b_ - a_) : (d_ - x) / (d_ - c_);
    }
    Interval support() const override { return {a_, d_}; }
    std::unique_ptr<IMembershipFunction> clone() const override { return std::make_unique<TrapezoidalMF>(*this); }
    double a() const noexcept { return a_; } double b() const noexcept { return b_; } double c() const noexcept { return c_; } double d() const noexcept { return d_; }
private:
    double a_, b_, c_, d_;
};

class GaussianMF final : public IMembershipFunction {
public:
    GaussianMF(double center, double sigma) : center_(center), sigma_(sigma) {
        if (sigma_ <= 0.0) throw std::invalid_argument("GaussianMF: sigma must be positive");
    }
    double evaluate(double x) const override {
        const double diff = (x - center_) / sigma_;
        return std::exp(-0.5 * diff * diff);
    }
    Interval support() const override { return {center_ - 4.0 * sigma_, center_ + 4.0 * sigma_}; }
    std::unique_ptr<IMembershipFunction> clone() const override { return std::make_unique<GaussianMF>(*this); }
    double center() const noexcept { return center_; } double sigma() const noexcept { return sigma_; }
private:
    double center_, sigma_;
};

class GeneralizedBellMF final : public IMembershipFunction {
public:
    GeneralizedBellMF(double a, double b, double c) : a_(a), b_(b), c_(c) {
        if (a_ <= 0.0) throw std::invalid_argument("GeneralizedBellMF: width 'a' must be positive");
    }
    double evaluate(double x) const override {
        const double d = std::abs((x - c_) / a_);
        return 1.0 / (1.0 + std::pow(d, 2.0 * b_));
    }
    Interval support() const override { return {c_ - 4.0 * a_, c_ + 4.0 * a_}; }
    std::unique_ptr<IMembershipFunction> clone() const override { return std::make_unique<GeneralizedBellMF>(*this); }
private:
    double a_, b_, c_;
};

class SigmoidalMF final : public IMembershipFunction {
public:
    SigmoidalMF(double a, double c) : a_(a), c_(c) {}
    double evaluate(double x) const override { return 1.0 / (1.0 + std::exp(-a_ * (x - c_))); }
    Interval support() const override {
        const double delta = (std::abs(a_) > 1e-9) ? 6.0 / std::abs(a_) : 10.0;
        return {c_ - delta, c_ + delta};
    }
    std::unique_ptr<IMembershipFunction> clone() const override { return std::make_unique<SigmoidalMF>(*this); }
private:
    double a_, c_;
};

class SingletonMF final : public IMembershipFunction {
public:
    explicit SingletonMF(double value, double tol = 1e-9) : value_(value), tol_(tol) {}
    double evaluate(double x) const override { return (std::abs(x - value_) <= tol_) ? 1.0 : 0.0; }
    Interval support() const override { return {value_ - tol_, value_ + tol_}; }
    std::unique_ptr<IMembershipFunction> clone() const override { return std::make_unique<SingletonMF>(*this); }
    double value() const noexcept { return value_; }
private:
    double value_, tol_;
};

inline std::shared_ptr<IMembershipFunction> makeTriangularMF(double a, double b, double c) { return std::make_shared<TriangularMF>(a, b, c); }
inline std::shared_ptr<IMembershipFunction> makeTrapezoidalMF(double a, double b, double c, double d) { return std::make_shared<TrapezoidalMF>(a, b, c, d); }
inline std::shared_ptr<IMembershipFunction> makeGaussianMF(double c, double s) { return std::make_shared<GaussianMF>(c, s); }
inline std::shared_ptr<IMembershipFunction> makeBellMF(double a, double b, double c) { return std::make_shared<GeneralizedBellMF>(a, b, c); }
inline std::shared_ptr<IMembershipFunction> makeSigmoidalMF(double a, double c) { return std::make_shared<SigmoidalMF>(a, c); }
inline std::shared_ptr<IMembershipFunction> makeSingletonMF(double v) { return std::make_shared<SingletonMF>(v); }

} // namespace fuzzy
} // namespace ga
