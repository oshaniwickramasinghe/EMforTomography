import math

def estimate_distance2(min_distance, min_median, max_distance, max_median, unknown_median):
    # Calculate the unknown distance using the proportional relationship
    unknown_distance = math.sqrt((min_distance**2 * unknown_median) / min_median)
    return unknown_distance

def estimate_distance(min_distance, min_median, max_distance, max_median, unknown_median):
    """
    Estimate the unknown distance based on the inverse proportionality between distance and median.

    Parameters:
    min_distance (float): Known minimum distance.
    min_median (float): Median value corresponding to the minimum distance.
    max_distance (float): Known maximum distance.
    max_median (float): Median value corresponding to the maximum distance.
    unknown_median (float): Median value for which the distance needs to be estimated.

    Returns:
    float: Estimated distance for the unknown median.
    """
    # Calculate the constant of proportionality using known values
    constant_min = min_distance * min_median
    constant_max = max_distance * max_median

    # Assuming the relationship is consistent, take the average constant
    constant = (constant_min + constant_max) / 2

    # Calculate the unknown distance using the inverse proportionality
    unknown_distance = constant / unknown_median

    return unknown_distance


# Example usage
if __name__ == "__main__":
    # Known values
    min_distance = 2
    min_median = 20
    max_distance = 8
    max_median = 5
    unknown_median = 8

    # Estimate the unknown distance
    estimated_distance = estimate_distance2(min_distance, min_median, max_distance, max_median, unknown_median)

    # Print the result
    print(f"Estimated distance for median {unknown_median}: {estimated_distance}")
