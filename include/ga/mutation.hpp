#pragma once

// Base mutation functionality
#include "ga/mutation/base_mutation.h"

// Binary representation mutation operators
#include "ga/mutation/bit_flip_mutation.h"

// Integer representation mutation operators
#include "ga/mutation/random_resetting_mutation.h"
#include "ga/mutation/creep_mutation.h"

// Real-valued representation mutation operators
#include "ga/mutation/uniform_mutation.h"
#include "ga/mutation/gaussian_mutation.h"

// Permutation representation mutation operators
#include "ga/mutation/swap_mutation.h"
#include "ga/mutation/inversion_mutation.h"
#include "ga/mutation/scramble_mutation.h"
#include "ga/mutation/insert_mutation.h"

// Adaptive and specialized mutation operators
#include "ga/mutation/self_adaptive_mutation.h"
#include "ga/mutation/list_mutation.h"
