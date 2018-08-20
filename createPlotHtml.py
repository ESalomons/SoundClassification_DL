import os


def doMain():
    # createPlotsInDir('plots/20180720_allOrgs')
    # createPlotsInDir('plots/20180816_orgsHelft2')
    createPlotsInDir('plots/20180817_orgStudio')

def createPlotsInDir(plotDirsLocation):
    nrPicsInRow = 3
    filenames = []
    for entry in os.walk(plotDirsLocation):
        filenames += entry[2]
    # filenames = [fn for fn in filenames if '.png' in fn and 'Gunshot' not in fn]
    filenames = [fn for fn in filenames if '.png' in fn]
    filenames = sorted(filenames)
    stepsize = len(filenames) // 3
    ziplist = zip(filenames[0:stepsize], filenames[stepsize:2*stepsize], filenames[2*stepsize:3*stepsize])

    htmlFilename = plotDirsLocation + '/overview.html'
    with open(htmlFilename, 'w') as htmlFile:
        titlename = (plotDirsLocation.split('/'))[-1]
        htmlFile.write('<html><head><title>' + titlename + '</title>\n')
        htmlFile.write('<body><H1>' + titlename + '</H1>\n')
        htmlFile.write('\n<table width="100%">')
        htmlFile.write('\n<STYLE TYPE="text/css">\n'
                       '<!--\n'
                       'TD{font-family: Arial; font-size: 16pt; font-weight:bold;}\n'
                       '--->\n'
                       '</STYLE>\n')
        body = ''
        for (one,two,three) in ziplist:
            body += '\n<tr>'
            body += '\n\t<td><img src="' + one \
                      + '" style="width:100%;"/></td>'
            body += '\n\t<td><img src="' + two \
                      + '" style="width:100%;"/></td>'
            body += '\n\t<td><img src="' + three \
                      + '" style="width:100%;"/></td>'
            body += '\n</tr>'

        htmlFile.write(body)
        htmlFile.write('\n</table>\n</body></html>' + '\n')




if __name__ == '__main__':
    doMain()
