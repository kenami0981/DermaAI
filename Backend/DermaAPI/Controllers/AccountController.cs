using Derma.Application.Account;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace Derma.API.Controllers
{
    public class AccountController : BaseApiController
    {
        [AllowAnonymous]
        [HttpPost("register")]
        public async Task<IActionResult> Register(AccountRegister.Command command)
        {
            var id = await Mediator.Send(command);
            return Ok(id);
        }
        [AllowAnonymous]
        [HttpPost("login")]
        public async Task<IActionResult> Login(AccountLogin.Command command)
        {
            var token = await Mediator.Send(command);

            if (token == null)
                return Unauthorized("Nieprawidłowy email lub hasło.");

            return Ok(new { token });
        }
    }
}