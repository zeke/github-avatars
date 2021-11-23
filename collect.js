const { chain } = require('lodash')
const words = require('an-array-of-english-words')
const usernames = chain(words).shuffle().filter(word => word.length < 6).value().slice(0, 100)
const download = require('download')

async function main () {
  console.log(usernames)
  for (const username of usernames) {
    console.log(username)
    await download(`https://github.com/${username}.png`, './avatars').catch(err => {
      // 404 who cares
    })
  }
}

main()