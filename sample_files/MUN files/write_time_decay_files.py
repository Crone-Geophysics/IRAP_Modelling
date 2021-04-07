def write_time_decay_files(channels, stations, fieldx, fieldy, fieldz, out_path):
	"""
	Write the text file in a format similar to a TEM file.
	:param channels: list of floats, channel times.
	:param stations: list of floats, station numbers.
	:param fieldx: 2D np array
	:param fieldy: 2D np array
	:param fieldz: 2D np array
	:param out_path: str, filepath of the output text file.
	"""
	channels = channels * 1e3  # Convert channel times to ms
	with open(out_path, 'w') as file:

		num_stations, num_channels = fieldx.shape
		station_names = [f"{i + 1}" for i in stations]
		station_names = [f"{i:^8}" for i in station_names]
		channel_names = [f"CH{i + 1}" for i in range(len(channels))]
		channel_names = [f"{i:^15}" for i in channel_names]

		file.write("Data type: dB/dt; UNIT: nT/s\n")
		file.write(f"Number of stations: {num_stations}\n")
		file.write(f"Stations (m): {' '.join([str(s) for s in stations]):^10}\n")
		file.write(f"Channel times (ms):\n")

		# Add the channel times
		for i in range(len(channels)):
			file.write(f"{i + 1:^8}{channels[i]:^ 8.4f}\n")

		file.write("EM data:\n")
		file.write(f"{'Station':^8}{'Component':^8}{''.join(channel_names)}\n")

		# Add the data
		print(f"Stations {stations.min()} - {stations.max()}")
		for i in range(num_stations):
			x = [f"{x:^ 15.5E}" for x in fieldx[i, :]]
			y = [f"{x:^ 15.5E}" for x in fieldy[i, :]]
			z = [f"{x:^ 15.5E}" for x in fieldz[i, :]]
			file.write(f"{station_names[i]:^8}{'X':^8}{''.join(x)}\n")
			file.write(f"{station_names[i]:^8}{'Y':^8}{''.join(y)}\n")
			file.write(f"{station_names[i]:^8}{'Z':^8}{''.join(z)}\n")