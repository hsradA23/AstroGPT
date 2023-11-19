<script>
  let arr = ["Hi, talk to your own AI yogi :)"];
  let prompt = "";

  let getData = () => {

    arr = [...arr, "You: " + prompt]

    fetch("/echo/" + prompt)
      .then((r) => r.text())
      .then((t) => {
        arr = [...arr, t.replaceAll('"', "")];
      });
  };
</script>

<div class="bg">
  <div class="msgs content">
    {#each arr as i}
      <div class="msg">
        {i}
      </div>
    {/each}
  </div>

  <div class="content">
    <div class="iarea">
      <input type="text" bind:value={prompt} on:submit={getData}/>
      <button on:click={getData}>Send</button>
    </div>
  </div>
</div>

<style>
  input {
    border-style: solid;
    border-radius: 10px;
    height: 5vh;
    width: 90%;
    background-color: #40414f;
    color: white;
    font-size: 2.5vh;
    padding: 0.1vh;
  }

  button{
    background-color: #397c65;
    border: none;
    padding: 0.1vh;
    height: 5vh;
    width: 10%;
    color: white;
    border-radius: 10px;
  }
  .bg {
    width: 100vw;
    height: 100vh;
    left: 0;
    top: 0;
    background-color: #444654;

    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }

  .content {
    padding: 1vh 20vh 0.5vh 20vh;
  }

  .msgs {
    display: flex;
    flex-direction: column;
    overflow-y: scroll;

  }

  .iarea {
    padding-bottom: 7%;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
  }

  .msg {
    color: white;
    padding: 1vh 2vw 1vh 2vw;
    border-bottom: 2px solid #6e7085; 
  }
</style>
