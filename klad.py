a = [(1,3), (3,4), (2,5)]
asort = sorted(a, key=lambda tup:tup[0])
print(list(zip(*asort)))

a = {'G428': {'20180816_orgsHelft1_100_20': [(1, 0.95), (2, 0.95), (3, 0.96), (4, 0.96), (5, 0.95), (1, 0.95), (2, 0.95), (3, 0.95), (4, 0.95), (5, 0.95), (1, 0.93), (2, 0.93), (3, 0.94), (4, 0.94), (5, 0.93), (1, 0.97), (2, 0.98), (3, 0.98), (4, 0.98), (5, 0.98)], '20180816_orgsHelft1_400_250_100_20': [(1, 0.95), (2, 0.95), (3, 0.96), (4, 0.95), (5, 0.95), (1, 0.94), (2, 0.95), (3, 0.95), (4, 0.95), (5, 0.95), (1, 0.94), (2, 0.93), (3, 0.93), (4, 0.95), (5, 0.93), (1, 0.96), (2, 0.97), (3, 0.98), (4, 0.96), (5, 0.97)], '20180816_orgsHelft1_400_300_200_100_50_20_10': [(1, 0.94), (2, 0.96), (3, 0.95), (4, 0.95), (5, 0.96), (1, 0.94), (2, 0.95), (3, 0.94), (4, 0.95), (5, 0.95), (1, 0.93), (2, 0.94), (3, 0.93), (4, 0.94), (5, 0.95), (1, 0.96), (2, 0.97), (3, 0.97), (4, 0.96), (5, 0.97)], '20180816_orgsHelft1_450_400_350_300_250_200_150_100_50_21': [(1, 0.96), (2, 0.94), (3, 0.95), (4, 0.93), (5, 0.96), (1, 0.95), (2, 0.94), (3, 0.95), (4, 0.92), (5, 0.96), (1, 0.94), (2, 0.92), (3, 0.93), (4, 0.92), (5, 0.94), (1, 0.98), (2, 0.98), (3, 0.98), (4, 0.94), (5, 0.98)]}, 'G527': {'20180816_orgsHelft1_100_20': [(1, 0.98), (2, 0.97), (3, 0.97), (4, 0.98), (5, 0.98), (1, 0.99), (2, 0.98), (3, 0.98), (4, 0.99), (5, 0.98), (1, 0.96), (2, 0.94), (3, 0.94), (4, 0.96), (5, 0.95), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0)], '20180816_orgsHelft1_400_250_100_20': [(1, 0.98), (2, 0.97), (3, 0.98), (4, 0.98), (5, 0.98), (1, 0.98), (2, 0.98), (3, 0.98), (4, 0.99), (5, 0.98), (1, 0.95), (2, 0.94), (3, 0.95), (4, 0.97), (5, 0.95), (1, 1.0), (2, 1.0), (3, 1.0), (4, 0.99), (5, 1.0)], '20180816_orgsHelft1_400_300_200_100_50_20_10': [(1, 0.98), (2, 0.98), (3, 0.98), (4, 0.98), (5, 0.98), (1, 0.98), (2, 0.99), (3, 0.98), (4, 0.99), (5, 0.99), (1, 0.95), (2, 0.96), (3, 0.95), (4, 0.96), (5, 0.96), (1, 0.99), (2, 1.0), (3, 0.99), (4, 0.99), (5, 1.0)], '20180816_orgsHelft1_450_400_350_300_250_200_150_100_50_21': [(1, 0.98), (2, 0.96), (3, 0.98), (4, 0.97), (5, 0.98), (1, 0.99), (2, 0.97), (3, 0.99), (4, 0.98), (5, 0.99), (1, 0.96), (2, 0.92), (3, 0.96), (4, 0.93), (5, 0.96), (1, 1.0), (2, 1.0), (3, 1.0), (4, 0.99), (5, 1.0)]}, 'Studio': {'20180816_orgsHelft1_100_20': [(1, 0.91), (2, 0.92), (3, 0.92), (4, 0.94), (5, 0.94), (1, 0.91), (2, 0.92), (3, 0.91), (4, 0.93), (5, 0.93), (1, 0.87), (2, 0.88), (3, 0.87), (4, 0.9), (5, 0.9), (1, 0.97), (2, 0.97), (3, 0.97), (4, 0.98), (5, 0.98)], '20180816_orgsHelft1_400_250_100_20': [(1, 0.92), (2, 0.93), (3, 0.93), (4, 0.94), (5, 0.94), (1, 0.92), (2, 0.93), (3, 0.92), (4, 0.93), (5, 0.94), (1, 0.88), (2, 0.89), (3, 0.89), (4, 0.9), (5, 0.9), (1, 0.97), (2, 0.97), (3, 0.97), (4, 0.98), (5, 0.98)], '20180816_orgsHelft1_400_300_200_100_50_20_10': [(1, 0.93), (2, 0.93), (3, 0.93), (4, 0.92), (5, 0.93), (1, 0.93), (2, 0.93), (3, 0.93), (4, 0.92), (5, 0.92), (1, 0.89), (2, 0.89), (3, 0.89), (4, 0.88), (5, 0.89), (1, 0.97), (2, 0.98), (3, 0.98), (4, 0.97), (5, 0.97)], '20180816_orgsHelft1_450_400_350_300_250_200_150_100_50_21': [(1, 0.93), (2, 0.94), (3, 0.93), (4, 0.94), (5, 0.93), (1, 0.92), (2, 0.93), (3, 0.92), (4, 0.94), (5, 0.92), (1, 0.88), (2, 0.9), (3, 0.89), (4, 0.91), (5, 0.89), (1, 0.97), (2, 0.97), (3, 0.98), (4, 0.98), (5, 0.98)]}}
