# Draw boundary
            # hull = ConvexHull(subData[["ratioAll", "ratioAll_Baseline"]].values)
            # # # Fill the area inside the convex hull
            # # plt.fill(subData[["ratioAll"]].values[hull.vertices], subData[["MTRPer"]].values[hull.vertices], color=color, alpha=0.2)
            # # # Plot the hull edges
            # # for simplex in hull.simplices:
            # #     plt.plot(subData[["ratioAll"]].values[simplex], subData[["MTRPer"]].values[simplex], color=color, alpha=0.2)
            # # Draw boundary in smooth
            # # Get the hull points
            # hullPoints = subData[["ratioAll", "ratioAll_Baseline"]].values[hull.vertices]
            # # Close the boundary by appending the first point to the end
            # hullPoints = np.vstack([hullPoints, hullPoints[0]])
            # # Create a B-spline representation of the hull points
            # tck, u = splprep(hullPoints.T, s=0.0001)  # s=0 means no smoothing
            # xSmooth, ySmooth = splev(np.linspace(0, 1, 100), tck)
            # # Fill the area inside the convex hull
            # plt.fill(xSmooth, ySmooth, color=color, alpha=0.2)
            # # Plot the smooth boundary
            # plt.plot(xSmooth, ySmooth, color=color, alpha=0.2)